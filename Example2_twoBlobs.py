import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph

from PowerRcut.PowerRcut import PRcut
from sklearn.cluster import k_means

from sklearn.cluster import spectral_clustering
from sklearn.manifold import spectral_embedding
"""
###################################################################################################
###################################### Basic Plots ################################################
###################################################################################################
"""

np.random.seed(10)

X1, y1 = make_blobs(n_samples = 200, centers=1, cluster_std = 4.0)
X2, y2 = make_blobs(n_samples = 1000, centers=1, cluster_std = 4.0)

X = np.concatenate([X1, X2], axis = 0)
y = np.ones(1100, dtype = np.int32)
y[100:] = 0


X_graph = kneighbors_graph(X, n_neighbors=40, include_self = False, mode = 'distance')
X_graph.data =  np.exp(-1*(X_graph.data**2.)/(2.*(X_graph.data.std()**2)))
X_graph = 0.5*(X_graph + X_graph.T)

# PRcut
embedPRcut, label_components = PRcut(X_graph, 3, nClust = 2)
c0, classPRcut, c2 = k_means(embedPRcut,2)

# Ncut
embedNcut = spectral_embedding(X_graph,n_components = 2, drop_first=True)
c0, classNcut, c2 = k_means(embedNcut,2)

# Plotting the results

from matplotlib import pyplot as plt

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
lim = [min(X[:,0])-0.01,max(X[:,0])+0.01,min(X[:,1])-0.01,max(X[:,1])+0.01]

plt.figure()
plt.scatter(X[:,0], X[:,1], color=colors[classPRcut].tolist())
plt.savefig("./img/Example2/PRcut.eps")
plt.close('all')

plt.figure()
plt.scatter(X[:,0], X[:,1], color=colors[classNcut].tolist())
plt.savefig("./img/Example2/Ncut.eps")
plt.close('all')

plt.figure()
plt.scatter(X[:,0], X[:,1])

plt.savefig("./img/Example2/original.eps")
plt.close('all')
