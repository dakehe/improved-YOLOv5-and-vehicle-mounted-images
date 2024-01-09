import numpy as np
import matplotlib.pyplot as plt

def mish(x):
    return x*np.tanh(np.log(1+np.exp(x)))

x = np.linspace(-5, 5, 100)
y = mish(x)

plt.plot(x, y)
plt.title('Mish Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()