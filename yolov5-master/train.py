# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
在自定义数据集上训练YOLOv5模型。

Models and datasets download automatically from the latest YOLOv5 release.
从最新的YOLOv5版本自动下载模型和数据集。
Models: https://github.com/ultralytics/yolov5/tree/master/models
Datasets: https://github.com/ultralytics/yolov5/tree/master/data
Tutorial: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (RECOMMENDED)
    $ python path/to/train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()  # 获取文件绝对路径
ROOT = FILE.parents[0]  # YOLOv5 root directory(ROOT得到YOLOv5根路径)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH(添加ROOT到系统路径)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# os.path.relpath是求相对路径的（把绝对路径转化成相对路径，前面那个相对于后面那个的相对路径）
# Path.cwd()获取当前路径，防止路径在ROOT前面，把这个补到后面weights等文件的路径中

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_version, check_yaml, colorstr, get_latest_run,
                           increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# local_rank和rank是一个意思，即代表第几个进程，world_size表示总共有n个进程
# 比如有2块gpu ,world_size = 5 , rank = 3,local_rank = 0 表示总共5个进程第 3 个进程内的第 1 块 GPU（不一定是0号gpu）。
# local_rank和rank的取值范围是从0到n-1


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')
    # 传入参数，callbacks（回调函数）

    # Directories（保存路径）
    # save_dir = runs/train/exp
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
     # parents：如果父目录不存在，是否创建父目录。
     # exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters（读预存超参数，用某颜色输出超参数）
    if isinstance(hyp, str):   # 如果hyp是str型格式
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict 引入了with语句来自动帮我们调用close()方法 读取配置文件中的参数
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    # 打印超参数

    # Save run settings（在exp下保存hyp.yaml\opt.yaml）
    if not evolve:
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)  # 将一个python值转换为yaml格式文件
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        # 日志记录器，一般是用wandb或者tensboard，观察数据的可视化
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config（多线程check数据）
    plots = not evolve and not opt.noplots  # create plots 是否需要画图: 所有的labels信息、前三次迭代的barch、训练结果等
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)  # 设置一系列的随机数种子
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    # 从coco128.yaml中取出train和val对应的地址
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes 数据集有多少种类别
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names 数据集所有类别的名字
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
    # 当前数据集是否是coco数据集(80个类别)  save_json和coco评价，如果是coco数据集会多执行一些操作

    # Model
    check_suffix(weights, '.pt')  # check weights，检查weights是不是以pt结尾
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):  # torch_distributed_zero_first函数的作用是只有主进程来加载数据，其他进程处于等待状态直到主进程加载完数据
            weights = attempt_download(weights)  # download if not found locally 如果没有pt就会尝试从官方下载
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak

        # ckpt: 模型的层

        # 这里加载模型有两种方式，一种是通过opt.cfg 另一种是通过ckpt['model'].yaml
        # 区别在于是否使用resume 如果使用resume会将opt.cfg设为空，按照ckpt['model'].yaml来创建模型
        # 这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
        # 原因: 保存的模型会保存anchors，有时候用户自定义了anchor之后，再resume，则原来基于coco数据集的anchor会自己覆盖自己设定的anchor
        # 详情参考: https://github.com/ultralytics/yolov5/issues/459
        # 所以下面设置intersect_dicts()就是忽略exclude

        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # 根据已有模型或者ckpt来创建新的模型
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        # csd就是原模型中的参数
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # 比较原模型中的参数与新模型的参数，把相同的保留
        model.load_state_dict(csd, strict=False)  # load 加载相同的参数到新模型
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:   # 不使用预训练
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP,返回true

    # list(model.named_parameters()) 权重层所有参数
    # 冻结权重层
    # 这里只是给了冻结权重层的一个例子, 但是不建议冻结权重层, 训练全部层参数, 可以得到更好的性能, 当然也会更慢
    # Freeze
    # 冻结的是models里面的yaml文件的网络层
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze  要冻结的参数名称（完整或部分）
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    # 确保模型的输入图片尺寸满足要求
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 模型默认的下采样倍率model.stride: [8, 16, 32]
    # gs代表模型下采样的最大步长: 后续为了保证输入模型的图片宽高是最大步长的整数倍
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    # 确保图片本身的尺寸满足要求，不足就会自动补成32的倍数

    # Batch size
    # 一般不执行，看自己的设置
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        # 确保batch size满足要求
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer（优化器）
    # nbs 标称的batch_size, 模拟的batch_size 比如默认的话上面设置的opt.batch_size = 16 -> nbs = 64
    # 也就是模型梯度累计 64/16=4(accumulate) 次之后就更新一次模型 等于变相的扩大了batch_size
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置超参: 权重衰减参数
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay（权重衰减）
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g = [], [], []  # optimizer parameter groups
    # 将模型参数分为三组(weights、biases、bn)来进行分组优化
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias所有的偏置参数
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)所有不待衰减的权重参数
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)所有待衰减的权重参数
            g[0].append(v.weight)

    # 选择优化器 并设置g2(bias)的优化方式
    if opt.optimizer == 'Adam':
        optimizer = Adam(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum将β1调整为动量
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else: # 用的sgd，也就是此处生效
        optimizer = SGD(g[2], lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 设置g0(weights)的优化方式
    optimizer.add_param_group({'params': g[0], 'weight_decay': hyp['weight_decay']})  # add g0 with weight_decay

    # 设置g1(bn)的优化方式
    optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g[1])} weight (no decay), {len(g[0])} weight, {len(g[2])} bias")
    del g
    # 删除三个变量 优化代码

    # Scheduler(学习率)
    if opt.cos_lr:  # 是否余弦学习率调整方式，这里用的余弦，1——0.01
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear（线性）
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    # 实例化

    # EMA（使用EMA（指数移动平均）对模型的参数做平均, 一种给予近期数据更高权重的平均方法, 以求提高测试指标并增加模型鲁棒。）
    # 也就是用于batchsize过小时，每次更新参数变化大，做个平均，使得参数变化更加平滑稳定
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume（恢复上次训练前的上下文），也用不到
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            # 恢复上次优化器的状态
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            # 恢复上次滑动平均的状态
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs
            # 恢复上次的epoch计数

        del ckpt, csd

    # DP mode
    # 是否使用DP mode
    # 如果rank=-1且gpu数量>1则使用DataParallel单机多卡模式  效果并不好（分布不平均）
    # 一般也用不到，而且现在更多是DDP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm 是否使用跨卡BN，也是多卡，用不到
    # 把模型复制到多卡里面，使不同卡之间数据做到一个同步
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    # 获取标签中最大类别值，与类别数作比较，如果小于类别数则表示有问题
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class（标签中最大类别值），coco里mlc是79
    nb = len(train_loader)  # number of batches（类别数），128/16=8
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    # '标签中最大类别值=mlc ，超过了 设置的类别数=nc in data. 可能的类标签是 nc'

    # Process 0 创建验证 dataloader和dataset
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:   # 如果不使用断点续训
            labels = np.concatenate(dataset.labels, 0) # 统计dataset的label信息,也就是所有图片里的所有labels
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:  # plots可视化数据集labels信息
                plot_labels(labels, names, save_dir)
                # 会在runs/train/exp下面画，留下labels.jpg和labels_correlogram.jpg

            # 计算默认锚框anchor与数据集标签框的高宽比
            # 标签的高h宽w与anchor的高h_a宽h_b的比值 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
            # 如果bpr小于98%，则根据k-mean算法聚类新的锚框
            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                #计算默认锚框anchor与数据集标签框的高宽比
            model.half().float()  # pre-reduce anchor precision
            # 预降锚精度

        callbacks.run('on_pretrain_routine_end')

    # DDP mode初始化多机多GPU的训练
    if cuda and RANK != -1:
        if check_version(torch.__version__, '1.11.0'):
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
        else:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model attributes初始化模型的超参数等模型属性
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)  检测层数(按hyps比例)，nl=3
    hyp['box'] *= 3 / nl  # scale to layers规模层
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    # 从训练样本标签得到类别权重（和类别中的目标数即类别频率成反比）
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names  # 获取类别名

    # Start training
    t0 = time.time()

    # 获取热身迭代的次数iterations, max(3 epochs, 1k iterations)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # warmup batch的数量
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1

    # 初始化maps(每个类别的map)和results
    maps = np.zeros(nc)  # mAP per class 每个类别的mAp，mean Average Precision, 即各类别AP的平均值
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    # 设置学习率衰减所进行到的轮次，即使打断训练，使用resume接着训练也能正常衔接之前的训练进行学习率衰减
    scheduler.last_epoch = start_epoch - 1  # do not move

    # 设置amp混合精度训练    GradScaler + autocast
    # 采用了AMP，可以加快运算 并且 减少显存占用。换句话说，可以节约时间，可以用更大的batch_size
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper = EarlyStopping(patience=opt.patience)

    # 初始化损失函数
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    # 开始训练
    print('开始啦！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        print('这是第', epoch, '轮')
        model.train()    #设置在训练模式下

        # Update image weights (optional, single-GPU only)并不一定好  默认是False的
        # 根据前面初始化的图片采样权重model.class_weights（每个类别的权重 频率高的权重小）以及maps配合每张图片包含的类别数
        # 通过rando.choices生成图片索引indices从而进行采用
        # Generate indices
        if opt.image_weights:   # 如果为True 进行图片采样策略(按数据集各类别权重采样)
            # 从训练(gt)标签获得每个类的权重  标签频率高的类权重低
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights类权重

            # 得到每一张图片对应的采样权重[128]
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights图片权重

            # random.choices: 从range(dataset.n)序列中按照weights(参考每张图片采样权重)进行采样, 一次取一个数字  采样次数为k
            # 最终得到所有图片的采样顺序(参考每张图片采样权重) list [128]
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx


        # Update mosaic border (optional)数据增强
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            # DDP模式打乱数据，并且dpp.sampler的随机采样数据是基于epoch+seed作为随机种子，每次epoch不同，随机种子不同
            train_loader.sampler.set_epoch(epoch)

        # 从train loader中读取样本
        # 迭代前信息提示
        pbar = enumerate(train_loader)  # 进度条，方便展示信息
        # 进度条标题
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))

        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar创建进度条
        optimizer.zero_grad() # 梯度清零

        # 开始批次内迭代
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)   ni: 计算当前迭代次数 iteration
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            # 把image分配到GPU上

            # Warmup
            # 热身训练（前nw次迭代）热身训练迭代的次数iteration范围[1:nw]  选取较小的accumulate，学习率以及momentum,慢慢的训练
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias的学习率从0.1下降到基准学习率lr*lf(epoch) 其他的参数学习率增加到lr*lf(epoch)
                    # lf为上面设置的余弦退火的衰减函数
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale 多尺度训练
            # 从[imgsz*0.5, imgsz*1.5+gs]间随机选取一个尺寸(32的倍数)作为当前batch的尺寸送入模型开始训练
            # imgsz: 默认训练尺寸   gs: 模型最大stride=32   [32 16 8]
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 下采样
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward 混合精度训练 开启autocast的上下文
            # 前向预测计算
            with torch.cuda.amp.autocast(amp):
                # pred: [8, 3, 68, 68, 25] [8, 3, 34, 34, 25] [8, 3, 17, 17, 25]
                # [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                pred = model(imgs)  # forward
                # 把imgs输入模型中，计算得出一个output，这里也就是pred

                # 计算损失，包括分类损失，置信度损失和框的回归损失
                # loss为总损失值  loss_items为一个元组，包含分类损失、置信度损失、框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                # 根据pred计算loss

                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode（DDP模式下设备之间的平均梯度）
                if opt.quad:
                    # 如果采用collate_fn4取出mosaic4数据loss也要翻4倍
                    loss *= 4.

            # Backward 反向传播  将梯度放大防止梯度的underflow（amp混合精度训练）
            # 反向梯度计算
            scaler.scale(loss).backward()
            # 从后往前推，计算各个的梯度

            # Optimize
            # 优化器梯度迭代
            # 模型反向传播accumulate次（iterations）后再根据累计的梯度更新一次参数
            if ni - last_opt_step >= accumulate:   #ni总迭代次数，last_opt_step=-1
                # scaler.step()首先把梯度的值unscale回来
                # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                scaler.step(optimizer)  # optimizer.step参数更新（权重在这里更新）

                scaler.update()     # 准备着，看是否要增大scaler
                optimizer.zero_grad()   # 梯度清零
                if ema:
                    ema.update(model)    # 当前epoch训练结束  更新ema
                last_opt_step = ni

            # Log
            # 打印Print一些信息 包括当前epoch、显存、损失(box、obj、cls、total)、当前batch的target的数量和图片的size等信息
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses更新平均损失
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # 进度条显示以上信息
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots)
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

            # # Plot
            # if plots and ni < 3:  # 将前三次迭代的barch的标签框再图片中画出来并保存  train_batch0/1/2.jpg
            #     f = save_dir / f'train_batch{ni}.jpg'  # filename
            #     Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
            #     if loggers['tb'] and ni == 0:  # TensorBoard
            #         with warnings.catch_warnings():
            #             warnings.simplefilter('ignore')  # suppress jit trace warning
            #             loggers['tb'].add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False),
            #                                             [])
            #
            # # wandb 显示信息
            # elif plots and ni == 10 and loggers['wandb']:
            #     wandb_logger.log({'Mosaics': [loggers['wandb'].Image(str(x), caption=x.name) for x in
            #         save_dir.glob('train*.jpg') if x.exists()]})


        # Scheduler  一个epoch训练结束后都要调整学习率（学习率衰减）
        # group中三个学习率（pg0、pg1、pg2）每个都要调整
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            # 将model中的属性赋值给ema
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

            # 判断当前epoch是否是最后一轮
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

            # noval: 是否只测试最后一轮  True: 只测试最后一轮   False: 每轮训练完都测试mAP
            if not noval or final_epoch:  # Calculate mAP
                # 测试使用的是ema（指数移动平均 对模型的参数做平均）的模型
                # results: [1] Precision 所有类别的平均precision(最大f1时)
                #          [1] Recall 所有类别的平均recall
                #          [1] map@0.5 所有类别的平均mAP@0.5
                #          [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
                #          [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
                # maps: [80] 所有类别的mAP@0.5:0.95

                results, maps, _ = val.run(data_dict, # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
                                           batch_size=batch_size // WORLD_SIZE * 2, # batch_size
                                           imgsz=imgsz, # test img size
                                           model=ema.ema,  # ema model
                                           single_cls=single_cls,  # 是否是单类数据集
                                           dataloader=val_loader,  # test dataloader
                                           save_dir=save_dir,    # 保存地址 runs/train/expn
                                           plots=False,  # 是否可视化
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)  # 损失函数(train)

            # Update best mAP 这里的best mAP其实是[P, R, mAP@.5, mAP@.5-.95]的一个加权值
            # fi: [P, R, mAP@.5, mAP@.5-.95]的一个加权值 = 0.1*mAP@.5 + 0.9*mAP@.5-.95
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            # 保存带checkpoint的模型用于inference或resuming training
            # 保存模型, 还保存了epoch, results, optimizer等信息
            # optimizer将不会在最后一轮完成后保存
            # model保存的是EMA的模型
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        # 日志: 打印训练时间
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        # 例如 5 epochs completed in 0.402 hours.

        # Strip optimizers
        # 模型训练完后, strip_optimizer函数将optimizer从ckpt中删除
        # 并对模型进行model.half() 将Float32->Float16 这样可以减少模型大小, 提高inference速度
        for f in last, best:  # 分别用最新模型和最好模型对训练结果进行评估
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    #例如 Validating runs\train\exp24\weights\best.pt...
                    results, _, _ = val.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, plots, epoch, results)

    torch.cuda.empty_cache()  # 释放显存
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='选择训练的权重文件')
    parser.add_argument('--cfg', type=str, default=ROOT / '../yolov5-master/models/yolov5s_gsconv_slim_c2f_det_spp.yaml', help='模型配置文件，例子：yolov5s.yaml')
    parser.add_argument('--data', type=str, default=ROOT / '../data/data.yaml', help='数据集配置文件，fruit.yaml所在位置') #data/coco128.yaml
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='初始超参文件')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=32, help='训练批次大小,total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=600, help='训练、测试图片分辨率大小')
    parser.add_argument('--rect', action='store_true', help='否采用矩形训练，默认False')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='是否接着上次的结果接着训练，默认False')
    parser.add_argument('--nosave', action='store_true', help='仅保存最后一个模型')
    parser.add_argument('--noval', action='store_true', help='是否只测试最后一轮 默认False  True: 只测试最后一轮   False: 每轮训练完都测试mAP')
    parser.add_argument('--noautoanchor', action='store_true', help='不自动调整anchor 默认False(自动调整anchor)')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='是否进行超参进化 默认False,evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='谷歌云盘bucket，一般不会用到,gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='使用加权图像选择进行训练')
    parser.add_argument('--device', default='', help='选择训练设备（GPUorCPU）,cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='是否进行多尺度训练 默认False,vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='数据集是否只有一个类别，默认False')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='是否使用跨卡同步BN,在DDP模式使用  默认False,use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='训练结果保存的根目录 默认是runs/train')
    parser.add_argument('--name', default='exp', help='训练结果保存的目录 默认是runs/train/exp')
    parser.add_argument('--exist-ok', action='store_true', help='如果文件不存在就新建或increment name  默认False(默认文件都是不存在的)')
    parser.add_argument('--quad', action='store_true', help='dataloader获取数据时, 是否使用collate_fn4代替collate_fn  默认False')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='标签平滑增强 默认0.0不增强  要增强一般就设为0.1')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='每一个“保存期”后的日志模型Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='rank为进程编号, -1且gpu=1时不进行分布式,DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()  # 检查yolov5的GitHub上代码是否更新
        check_requirements(exclude=['thop'])  # 检查requirements中的包是否安装成功

    # Resume 是否接着上次的结果接着训练，默认False
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        # 要么告诉它cfg（网格结构），要么告诉它weights（网格权重），不然就会报错提示，也就是存不存在这两个文件
        if opt.evolve: # 是否进行超参进化 默认False
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        # 以增量路径保存，在这里是指在runs下以exp，exp2...等路径保存，训练结果保存的目录 默认是runs/train/exp

    # DDP mode（DistributedDataParallel），多卡式训练
    # 选择gpu还是cpu，分布式训练等，没用到
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # Evolve hyperparameters (optional)
    # 自动调参，进化超参数，需要的时间和资源非常庞大，一般用不到，而且手动调参效果也还可以
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参数进化列表,括号里分别为(突变规模, 最小值,最大值)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3) 学习率
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf) 余弦退火超参数
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1 学习率动量
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay 权重衰减系数
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
