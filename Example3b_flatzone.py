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

sizePRcut = []
sizeRcut = []
sizeBlack = []


for i in tqdm(range(0,100,2)):
    x, y = np.indices((200*unit, 400*unit))
    img = np.zeros(x.shape, dtype=np.int32)
    y1 = 50 + i
    y2 = y1 + 255
    img[np.where(y < y1)] = 0
    img[np.where(y > y2)] = 255
    img[np.where(np.logical_and(y>=y1, y<=y2))] = y[np.where(np.logical_and(y>=y1, y<=y2))] - y1
    img = img.reshape(200, 400, 1)


    # Construct the graph
    beta = 1.
    eps = 1e-6
    s0,s1,s2 = img.shape
    graph, graph_Bucket = method1(img,s0,s1,s2,spat_distance = 1, beta = beta, eps = eps)
    graph = 0.5 * (graph + graph.T)


    # PRcut
    embedPRcut, label_components = PRcut(graph, 2, nClust = 2)
    c0, classPRcut, c2 = k_means(embedPRcut,2)


    # Rcut
    L = laplacian(graph)
    eigval, embedRcut = eigsh(L, 2, sigma = 1e-6)
    d0, classRcut, d2 = k_means(embedRcut,2)
    # classRcut = classRcut.reshape(img.shape[:2])



    sizeBlack.append(y1)
    sPRcut = float(len(np.where(classPRcut==classPRcut[0])[0]) - y1*200)/(s0*s1 + 0.0)
    sizePRcut.append(sPRcut)
    sRcut = float(len(np.where(classRcut==classRcut[0])[0]) - y1*200)/(s0*s1 + 0.0)
    sizeRcut.append(sRcut)


plt.figure()
plt.plot(sizeBlack, sizePRcut, label = 'PRcut')
plt.plot(sizeBlack, sizeRcut, label = 'Rcut')
plt.legend()
plt.ylabel('propotional size of the boundary allotted to black cluster')
plt.xlabel('size of the initial cluster')
plt.savefig('./img/Example3/PRcutvsRcut.eps')
