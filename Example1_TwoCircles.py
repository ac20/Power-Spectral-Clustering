import numpy as np
from sklearn.datasets import make_circles
from sklearn.neighbors import kneighbors_graph

from PowerRcut.PowerRcut import PRcut
from sklearn.cluster import k_means

from sklearn.cluster import spectral_clustering

from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
import scipy as sp
from tqdm import tqdm
import time
"""
###################################################################################################
###################################### Basic Plots ################################################
###################################################################################################
"""

np.random.seed(10)

st_time = time.clock()
X, y = make_circles(n_samples = 1500, factor = 0.6, noise = 0.07)
X_graph = kneighbors_graph(X, n_neighbors=40, include_self = False, mode = 'distance')
X_graph.data =  np.exp(-1*X_graph.data/X_graph.data.std())
X_graph = 0.5*(X_graph + X_graph.T)
print 'Time taken to construct the graph : ', time.clock() - st_time

# Power Ratio cut
st_time = time.clock()
embedPRcut, tmp = PRcut(X_graph, 2, nClust = 2)
c0, classPRcut, c2 = k_means(embedPRcut,2)
print '             Time taken for PRcut : ', time.clock() - st_time


# Ratio cut
st_time = time.clock()
L = laplacian(X_graph)
eigval, embedRcut = eigsh(L, 2, sigma = 1e-6)
d0, labelsRcut, d2 = k_means(embedRcut,2)
print '              Time taken for Rcut : ', time.clock() - st_time



# Ratio cuts for low noise

np.random.seed(10)
X, y = make_circles(n_samples = 1500, factor = 0.6, noise = 0.05)
X_graph = kneighbors_graph(X, n_neighbors=40, include_self = False, mode = 'distance')
X_graph.data =  np.exp(-1*X_graph.data/X_graph.data.std())
X_graph = 0.5*(X_graph + X_graph.T)
L = laplacian(X_graph)
eigval, embedRcut = eigsh(L, 2, sigma = 1e-6)
d0, labelsRcutLowNoise, d2 = k_means(embedRcut,2)

# Plotting the results

from matplotlib import pyplot as plt

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

plt.figure()
plt.scatter(X[:,0], X[:,1], color=colors[classPRcut].tolist())
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.savefig("./img/Example1/PRcut.eps")
plt.close('all')

plt.figure()
plt.scatter(X[:,0], X[:,1], color=colors[labelsRcut].tolist())
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.savefig("./img/Example1/Rcut.eps")
plt.close('all')


plt.figure()
plt.scatter(X[:,0], X[:,1], color=colors[labelsRcutLowNoise].tolist())
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.savefig("./img/Example1/RcutLowNoise.eps")
plt.close('all')

plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.savefig("./img/Example1/original.eps")
plt.close('all')



"""
###################################################################################################
#################################### Noise Analysis ###############################################
###################################################################################################
"""


from tqdm import tqdm

from sklearn.metrics import adjusted_mutual_info_score as AMI

print 'Calculating the noise robustness...'

l_x = np.arange(0.02,0.15,0.002)
l_yPRcut = []
l_yRcut = []
for noise in tqdm(l_x):
    a_PRcutTmp = []
    a_RcutTmp = []
    for _ in range(1):
        np.random.seed(10)
        X, y = make_circles(n_samples = 1500, factor = 0.5, noise = noise)
        X_graph = kneighbors_graph(X, n_neighbors=40, include_self = False, mode = 'distance')
        X_graph.data =  np.exp(-1*X_graph.data/X_graph.data.std())
        X_graph = 0.5*(X_graph + X_graph.T)

        # PRcut
        embedPRcut, tmp = PRcut(X_graph, 2, nClust = 2)
        c0, classPRcut, c2 = k_means(embedPRcut,2)
        a_PRcutTmp.append(AMI(y,classPRcut))
        # Rcut
        L = laplacian(X_graph)
        eigval, embedRcut = eigsh(L, 2, sigma = 1e-6)
        d0, classRcut, d2 = k_means(embedRcut,2)
        a_RcutTmp.append(AMI(y,classRcut))
    l_yPRcut.append(np.mean(a_PRcutTmp))
    l_yRcut.append(np.mean(a_RcutTmp))



plt.figure()
plt.axis([0.02,0.10,0,1])
plt.plot(l_x, np.array(l_yPRcut), 'b*-' , label="PRcut")
plt.plot(l_x, np.array(l_yRcut), 'ro-', label="Ratio cut")
plt.legend()
plt.xlabel('Noise')
plt.ylabel('Adjusted Mutual Information')
plt.savefig("./img/Example1/NoiseAnalysis.eps")
plt.close('all')
