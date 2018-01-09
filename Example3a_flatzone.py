import numpy as np
from PowerRcut.PowerRcut import PRcut
from PowerRcut.img_to_graph.img_to_graph import *
from matplotlib import pyplot as plt
from sklearn.cluster import k_means

from sklearn.cluster import spectral_clustering
from sklearn.manifold import spectral_embedding

from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
import scipy as sp
from tqdm import tqdm
import time

unit = 1

x, y = np.indices((200*unit, 400*unit))
img = np.zeros(x.shape, dtype=np.int32)
y1 = 140 # Change this parameter to observe the differences in the size of the black cluster.
y2 = y1 + 255
img[np.where(y < y1)] = 0
img[np.where(y > y2)] = 255
img[np.where(np.logical_and(y>=y1, y<=y2))] = y[np.where(np.logical_and(y>=y1, y<=y2))] - y1
img = img.reshape(200, 400, 1)

plt.imsave('./img/Example3/original.eps', img.reshape(200,400), cmap=plt.cm.gray)



beta = 1.
eps = 1e-6
s0,s1,s2 = img.shape
graph, graph_Bucket = method1(img,s0,s1,s2,spat_distance = 1, beta = beta, eps = eps)
graph = 0.5 * (graph + graph.T)


# Power Ratio cut
st_time = time.clock()
embedPRcut, label_components = PRcut(graph, 2, nClust = 2)
c0, classPRcut, c2 = k_means(embedPRcut,2)
classPRcut = classPRcut.reshape(img.shape[:2])
print '             Time taken for PRcut : ', time.clock() - st_time
print '            Size of black cluster : ', float(len(np.where(classPRcut==classPRcut[0])[0]))

# Ratio cut
st_time = time.clock()
L = laplacian(graph)
eigval, embedRcut = eigsh(L, 2, sigma = 1e-6)
d0, labelsRcut, d2 = k_means(embedRcut,2)
labelsRcut = labelsRcut.reshape(img.shape[:2])
print '              Time taken for Rcut : ', time.clock() - st_time
print '            Size of black cluster : ', float(len(np.where(labelsRcut==labelsR`cut[0])[0]))


img[:,-1,:] = 1
img[-1,:,:] = 1
img[:,0,:] = 1
img[0,:,:] = 1

plt.figure()
plt.imshow(img.reshape(200,400), cmap=plt.cm.gray)
plt.axis('off')
plt.savefig('./img/Example3/original.eps')
plt.close('all')

plt.figure()
plt.imshow(img.reshape(200,400), cmap=plt.cm.gray)
for i in range(2):
    plt.contour(classPRcut == i, contours=1,colors=[plt.cm.spectral(i / float(2))])
plt.axis('off')
plt.savefig('./img/Example3/PRcut.eps')
plt.close('all')


plt.figure()
plt.imshow(img.reshape(200,400), cmap=plt.cm.gray)
for i in range(2):
    plt.contour(labelsRcut == i, contours=1,colors=[plt.cm.spectral(i / float(2))])
plt.axis('off')
plt.savefig('./img/Example3/Rcut.eps')
plt.close('all')
