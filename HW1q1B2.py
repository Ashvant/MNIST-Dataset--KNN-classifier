import mnist
import matplotlib.pyplot as plt
import numpy as np
from random import *

x_train, t_train, x_test, t_test = mnist.load()
X = 255-x_train
plt.gray()
i=0
while i<15:
    i=i+1
    plt.subplot(3,5,i)
    plt.imshow((X[randint(1,60000)]).reshape((28,28)))
plt.show()