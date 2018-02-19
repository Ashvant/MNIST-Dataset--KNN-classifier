import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
np.set_printoptions(threshold=np.inf)
x_train, t_train, x_test, t_test = mnist.load()
X = x_train
y = t_train
data_size = []
scores_training = []
scores_test = []
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
neigh = KNeighborsClassifier(n_neighbors=5,n_jobs=4)
neigh.fit(X,y)
scores2 = neigh.predict(x_test[:10000])
print(scores2)
print(t_test)
bool_idx=[scores2 != t_test]
print(bool_idx)
print(x_test[bool_idx])
misclassified = x_test[bool_idx]
k_images = neigh.kneighbors(misclassified[0].reshape(1,-1), n_neighbors=5, return_distance=False)
print(k_images)
for j in range(5):
        plt.subplot(1, 5,j+1)
        plt.imshow((X[k_images[0][j]]).reshape((28,28)))
plt.show()