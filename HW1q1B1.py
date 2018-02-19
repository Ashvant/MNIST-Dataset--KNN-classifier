import mnist
import matplotlib.pyplot as plt
import numpy as np

x_train, t_train, x_test, t_test = mnist.load()
X = 255-x_train
plt.gray()
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow((X[i]).reshape((28,28)))
plt.show()