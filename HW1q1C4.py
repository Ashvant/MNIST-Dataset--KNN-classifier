import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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
for k in range (5000,60001,5000):
    data_size.append(k)
    neigh = KNeighborsClassifier(n_neighbors=3,n_jobs=4)
    neigh.fit(X[:k],y[:k])
    scores2 = neigh.score(x_test[:10000],t_test[:10000])
    print(scores2)
    scores_test.append(1-scores2)
plt.plot(data_size,scores_test,label="test error")
plt.xlabel('Training Data size')
plt.ylabel('error')
plt.show()