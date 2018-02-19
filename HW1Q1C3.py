import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x_train, t_train, x_test, t_test = mnist.load()
X = x_train
y = t_train
neighbours = []
scores_training = []
scores_test = []
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
for k in range (1,10000,200):
    neighbours.append(1/k)
    neigh = KNeighborsClassifier(n_neighbors=k,n_jobs=4)
    neigh.fit(X[:60000],y[:60000])
    scores1 = neigh.score(X[:10000],y[:10000])
    print (scores1)
    scores_training.append(1-scores1)
    scores2 = neigh.score(x_test[:10000],t_test[:10000])
    print(scores2)
    scores_test.append(1-scores2)
plt.plot(neighbours,scores_training,label="training error")
plt.plot(neighbours,scores_test,label="test error")
plt.xlabel('i/k')
plt.ylabel('error')
plt.show()