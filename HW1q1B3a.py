import mnist
import matplotlib.pyplot as plt
import numpy as np
from random import *

x_train, t_train, x_test, t_test = mnist.load()
X = 255-x_train
plt.gray()
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow((X[i]).reshape((28,28)))
print((t_train[0:100]).reshape(10,10))
plt.show()