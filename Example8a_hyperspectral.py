import pandas as pd
import numpy as np
from PowerRcut.PowerRcut_hyperspectral import PRcut
from PowerRcut.img_to_graph.img_to_graph import *
from matplotlib import pyplot as plt
from sklearn.cluster import k_means
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
import scipy as sp
from tqdm import tqdm
import time


from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from scipy.io import loadmat



"""
Calculating the timing statistics for different subimages of University Pavia Dataset
-------------------------------------------------------------------------------------
"""

arr_scale = []
arr_processtime = []
arr_Rcut_time = []
arr_accRcut = []
arr_PRcut_time = []
arr_accPRcut = []

for scale in np.arange(0.7,0.99,0.1):
    # Need only construct the graph once per scale. (Much faster this way)
    st_time = time.clock()
    img = loadmat("./Hyperspectral_data/PaviaU.mat")['paviaU']
    s0,s1,s2 = img.shape
    img = np.array(img[-int(scale*s0):,:int(scale*s1),:],np.int32)

    img_gt = loadmat("./Hyperspectral_data/PaviaU_gt.mat")['paviaU_gt']
    img_gt = img_gt[-int(scale*s0):,:int(scale*s1)]
    s0,s1,s2 = img.shape

    # Constructing the graph from the data using the PowerRcut.img_to_graph module.
    graph, w = method1(img,s0,s1,s2,spat_distance = 1, beta = 5., eps = 1e-6)
    graph = 0.5 * (graph + graph.T)
    process_time = time.clock() - st_time

    for _ in range(10):
        # Performing spectral clustering on the data
        st_time = time.clock()
        L = laplacian(graph)
        eigval, embedRcut = eigsh(L, 20, sigma = 1e-6 )
        d0, labelsRcut, d2 = k_means(embedRcut,50)
        Rcut_time = time.clock() - st_time

        # Calculating the accuracy for spectral clustering
        # Each class obtained on clustering is assumed to belong to the largest intersection
        z = confusion_matrix(img_gt.flatten(),labelsRcut)
        accRcut = 0
        for i in range(z.shape[0]):
            accRcut += z[np.argmax(z[:,i]),i]
        accRcut = (accRcut+0.0)/(s0*s1)

        # Performing PRcut on the data
        st_time = time.clock()
        embedPRcut, label_components = PRcut(graph, 20, discrete_Method = 'kmeans', nClust=7, threshComp=200, verbose=True)
        d0, labelsPRcut, d2 = k_means(embedPRcut,50)
        PRcut_time = time.clock() - st_time

        # Calculating the accuracy for PRcut
        # Each class obtained on clustering is assumed to belong to the largest intersection
        z = confusion_matrix(img_gt.flatten(),labelsPRcut)
        accPRcut = 0
        for i in range(z.shape[0]):
            accPRcut += z[np.argmax(z[:,i]),i]
        accPRcut = (accPRcut+0.0)/(s0*s1)

        arr_scale.append(scale)
        arr_processtime.append(process_time)
        arr_Rcut_time.append(Rcut_time)
        arr_accRcut.append(accRcut)
        arr_PRcut_time.append(PRcut_time)
        arr_accPRcut.append(accPRcut)

        print '***********************************'
        print '                   Scale = ', img.shape
        print 'Constructing graph time  = ', process_time
        print 'Ratio cut          time  = ', Rcut_time
        print '               Accuracy  = ', AMI(img_gt.flatten(),labelsRcut), ARI(img_gt.flatten(),labelsRcut),accRcut
        print 'PowerRatio cut     time  = ', PRcut_time
        print '               Accuracy  = ', AMI(img_gt.flatten(),labelsPRcut), ARI(img_gt.flatten(),labelsPRcut),accPRcut
        print '***********************************'


# Writing the results to a csv file
f = open('results_8a.csv', "w+")
for i in range(len(arr_scale)):
    l = ""
    l += str(arr_scale[i])+","+str(arr_processtime[i])+","
    l += str(arr_Rcut_time[i])+","+str(arr_accRcut[i])+","+str(AMI(img_gt.flatten(),labelsRcut))+","+str(ARI(img_gt.flatten(),labelsRcut))+","
    l += str(arr_PRcut_time[i])+","+str(arr_accPRcut[i])+","+str(AMI(img_gt.flatten(),labelsPRcut))+","+str(ARI(img_gt.flatten(),labelsPRcut))
    f.write(l)
    f.write("\n")
f.close()



"""
Recording an typical solution
-----------------------------
"""

st_time = time.clock()
img = loadmat("./Hyperspectral_data/PaviaU.mat")['paviaU']
img = img[-200:,:200,:]
img = np.array(img, dtype = np.int32)
img_gt = loadmat("./Hyperspectral_data/PaviaU_gt.mat")['paviaU_gt']
img_gt = img_gt[-200:,:200]

print '... Constructing graph'
s0,s1,s2 = img.shape
graph, w = method1(img,s0,s1,s2,spat_distance = 1, beta = 5., eps = 1e-6)
graph = 0.5 * (graph + graph.T)

process_time = time.clock() - st_time

st_time = time.clock()
L = laplacian(graph, normed = True)
eigval, embedRcut = eigsh(L, k=20, sigma = 1e-6 )
d0, labelsRcut, d2 = k_means(embedRcut,50)
Rcut_time = time.clock() - st_time

z = confusion_matrix(img_gt.flatten(),labelsRcut)
accRcut = 0
for i in range(z.shape[0]):
    accRcut += z[np.argmax(z[:,i]),i]
accRcut = (accRcut+0.0)/(s0*s1)

st_time = time.clock()
embedPRcut, label_components = PRcut(graph, 20, discrete_Method = 'kmeans', nClust=7, threshComp=200, verbose=True)
d0, labelsPRcut, d2 = k_means(embedPRcut,50)
PRcut_time = time.clock() - st_time


z = confusion_matrix(img_gt.flatten(),labelsPRcut)
accPRcut = 0
for i in range(z.shape[0]):
    accPRcut += z[np.argmax(z[:,i]),i]
accPRcut = (accPRcut+0.0)/(s0*s1)


print '***********************************'
print '                   Scale = ', img.shape
print 'Constructing graph time  = ', process_time
print 'Ratio cut          time  = ', Rcut_time
print '               Accuracy  = ', AMI(img_gt.flatten(),labelsRcut), ARI(img_gt.flatten(),labelsRcut),accRcut
print 'PowerRatio cut     time  = ', PRcut_time
print '               Accuracy  = ', AMI(img_gt.flatten(),labelsPRcut), ARI(img_gt.flatten(),labelsPRcut),accPRcut
print '***********************************'


plt.imsave("./img/paviaU_gt.pdf",img_gt)
plt.imsave("./img/paviaU_PRcut.pdf",np.array(labelsPRcut).reshape(img_gt.shape))
plt.imsave("./img/paviaU_Rcut.pdf",np.array(labelsRcut).reshape(img_gt.shape))

from scipy.ndimage.filters import gaussian_filter

filtered_PRcut = gaussian_filter(labelsPRcut.reshape(img_gt.shape), 2)
plt.imsave("./img/paviaU_PRcutFiltered.pdf",filtered_PRcut)

from process_small_clusters import *
embedPRcut_filtered = process_small_clusters(embedPRcut,label_components,graph,200)
d0, labelsPRcut_filtered, d2 = k_means(embedPRcut_filtered,20)
z = confusion_matrix(img_gt.flatten(),labelsPRcut_filtered)
accPRcut_filtered = 0
for i in range(z.shape[0]):
    accPRcut_filtered += z[np.argmax(z[:,i]),i]
accPRcut_filtered = (accPRcut_filtered+0.0)/(s0*s1)
plt.imsave("./img/paviaU_embedFilter.pdf",np.array(labelsPRcut_filtered).reshape(img_gt.shape))
