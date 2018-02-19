import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x_train, t_train, x_test, t_test = mnist.load()
plt.gray()
neigh = KNeighborsClassifier(n_neighbors=5)
X = 255-x_train
y = t_train
x_test = 255 - x_test
neigh.fit(X[:60000],y[:60000])
k_images = neigh.kneighbors(X=x_test[0:10], n_neighbors=5, return_distance=False)
print(k_images)
for i in range(10):
    for j in range(5):
        plt.subplot(10, 5, (i*5)+(j+1))
        plt.imshow((X[k_images[i][j],:]).reshape((28,28)))
plt.show()