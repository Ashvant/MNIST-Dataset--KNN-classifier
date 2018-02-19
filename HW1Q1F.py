import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math
from scipy.spatial.distance import directed_hausdorff
np.set_printoptions(threshold=np.inf)
x_train, t_train, x_test, t_test = mnist.load()
X = x_train
y = t_train
neighbours = []
scores_training = []
scores_test = []
def customWeights(a):
    a = [8,4,2]
    return a
neigh = KNeighborsClassifier(n_neighbors=3,metric="euclidean",weights=customWeights)
neigh.fit(X[:60000],y[:60000])
scores2 = neigh.score(x_test[:10000],t_test[:10000])
print("Eucledian error -- " ) 
print(1-scores2)
neigh = KNeighborsClassifier(n_neighbors=3,metric="minkowski",p=1,weights = customWeights)
neigh.fit(X[:60000],y[:60000])
scores2 = neigh.score(x_test[:10000],t_test[:10000])
print("Manhattan error --")
print(1-scores2)
neigh = KNeighborsClassifier(n_neighbors=3,metric="chebyshev")
neigh.fit(X[:60000],y[:60000])
scores2 = neigh.score(x_test[:10000],t_test[:10000])
print("Chebyshev error --")
print(1-scores2)