import numpy as np
from PowerRcut.PowerRcut import PRcut
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph

from matplotlib import pyplot as plt
from sklearn.cluster import k_means

from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
import scipy as sp
from tqdm import tqdm
import time
import timeit



size_Samples = range(100,9000,200)
np.random.seed(10)
no_repeat = 10

time_PRcut = np.zeros((len(size_Samples),no_repeat))
time_Rcut = np.zeros((len(size_Samples),no_repeat))
time_index = np.zeros((len(size_Samples),no_repeat))
for i in tqdm(range(len(size_Samples))):
    n_samples = size_Samples[i]
    for j in range(no_repeat):
        time_index[i,j] = size_Samples[i]
        X, y = make_blobs(n_samples = n_samples, centers=2, cluster_std = 4.0)
        X_graph = kneighbors_graph(X, n_neighbors=40, include_self = False, mode = 'distance')
        X_graph.data =  np.exp(-1*(X_graph.data**2.)/(2.*(X_graph.data.std()**2)))
        X_graph = 0.5*(X_graph + X_graph.T)

        # PRcut
        st_time = time.clock()
        embedPRcut, label_components = PRcut(X_graph, 2, nClust = 2)
        tmp = time.clock() - st_time
        time_PRcut[i,j] = tmp

        # Rcut
        st_time = time.clock()
        L = laplacian(X_graph)
        eigval, embedRcut = eigsh(L, 2, sigma = 1e-6)
        tmp = time.clock() - st_time
        time_Rcut[i,j] = tmp

plt.figure()
plt.plot(size_Samples, np.mean(time_Rcut, axis = 1), label="Rcut")
plt.plot(size_Samples, np.mean(time_PRcut, axis = 1), label="PRcut")
plt.xlabel('size')
plt.ylabel('time')
plt.legend()
plt.savefig('./img/Example4.eps')

from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X)

clf = LR()
a =  time_index.flatten()
a = a.reshape((a.shape[0],1))
a = poly.fit_transform(a)

plt.figure()
plt.scatter(time_index.flatten(), time_Rcut.flatten(), color='b', label="Ratio cut")
clf.fit(a, time_Rcut.flatten())
plt.plot(time_index.flatten(),clf.predict(a),'b')

plt.scatter(time_index.flatten(), time_PRcut.flatten(), color='r', label="Power Ratio cut")
clf.fit(a, time_PRcut.flatten())
plt.plot(time_index.flatten(),clf.predict(a),'r')
plt.xlabel('size')
plt.ylabel('time')
plt.legend()
plt.savefig('./img/Example4Scatter.eps')