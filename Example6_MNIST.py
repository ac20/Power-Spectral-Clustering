import pandas as pd
import numpy as np
from PowerRcut.PowerRcut import PRcut
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

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

from sklearn.decomposition import PCA



"""
Calculating the timing statistics for MNIST dataset
---------------------------------------------------
"""

arr_dimData = []
arr_sizeData = []
arr_process_time = []
arr_Rcut_time, arr_accRcut, arr_AMIRcut, arr_ARIRcut = [], [], [], []
arr_PRcut_time, arr_accPRcut, arr_AMIPRcut, arr_ARIPRcut = [], [], [], []

for thresh in [0.1, 0.2, 0.3]:
    for dim in [10,15,20]:
        np.random.seed(10)
        data = pd.read_csv("./Data/MNIST/train.csv")
        y = np.array(data['label'])
        X = np.array(data.drop(['label'], axis = 1), dtype = np.float64)

        m = np.random.random(X.shape[0])

        ym = y[m < thresh]
        Xm = X[m < thresh,:]

        st_time = time.clock()
        pca = PCA(n_components=dim)
        X_reduced = pca.fit_transform(Xm)
        X_graph = kneighbors_graph(X_reduced, n_neighbors=20, include_self = False, mode = 'distance')
        n_components, labels = connected_components(X_graph)
        print n_components
        X_graph.data =  np.exp(-1*(X_graph.data)/((X_graph.data.std()))) + 1e-6
        X_graph = 0.5*(X_graph + X_graph.T)
        process_time = time.clock() - st_time

        # Rcut
        st_time = time.clock()
        L = laplacian(X_graph)
        eigval, embedRcut = eigsh(L, 10, sigma = 1e-6)
        d0, labelsRcut, d2 = k_means(embedRcut,10)
        Rcut_time = time.clock() - st_time

        # Calculating the accuracy for Rcut
        # Each class obtained on clustering is assumed to belong to the largest intersection
        z = confusion_matrix(ym,labelsRcut)
        accRcut = 0
        for i in range(z.shape[0]):
            accRcut += z[np.argmax(z[:,i]),i]
        accRcut = (accRcut+0.0)/len(ym)

        # PRcut
        st_time = time.clock()
        embedPRcut, label_components = PRcut(X_graph, 10, discrete_Method='kmeans', nClust=2, verbose = True)
        d0, labelsPRcut, d2 = k_means(embedPRcut,10)
        PRcut_time = time.clock() - st_time

        # Calculating the accuracy for PRcut
        # Each class obtained on clustering is assumed to belong to the largest intersection
        z = confusion_matrix(ym,labelsPRcut)
        accPRcut = 0
        for i in range(z.shape[0]):
            accPRcut += z[np.argmax(z[:,i]),i]
        accPRcut = (accPRcut+0.0)/len(ym)

        print '********************************************'
        print '              Dimension = ', dim
        print '                   Size = ', Xm.shape[0]
        print 'time for preprocessing  = ', process_time
        print 'Rcut  :      time_taken = ', Rcut_time
        print '               Accuracy = ', accRcut
        print '                    ARI = ', ARI(ym,labelsRcut)
        print '                    AMI = ', AMI(ym,labelsRcut)
        print 'PRcut :      time_taken = ', PRcut_time
        print '               Accuracy = ', accPRcut
        print '                    ARI = ', ARI(ym,labelsPRcut)
        print '                    AMI = ', AMI(ym,labelsPRcut)
        print '********************************************'


        # General_Param
        arr_dimData.append(dim)
        arr_sizeData.append(Xm.shape[0])
        arr_process_time.append(process_time)
        # Rcut Results
        arr_Rcut_time.append(Rcut_time)
        arr_accRcut.append(accRcut)
        arr_AMIRcut.append(ARI(ym,labelsRcut))
        arr_ARIRcut.append(AMI(ym,labelsRcut))
        # PRcut Results
        arr_PRcut_time.append(PRcut_time)
        arr_accPRcut.append(accPRcut)
        arr_AMIPRcut.append(ARI(ym,labelsPRcut))
        arr_ARIPRcut.append(AMI(ym,labelsPRcut))

f = open('results_6.csv', "w+")
for i in range(len(arr_dimData)):
    l = ""
    l += str(arr_dimData[i])+","+str(arr_sizeData[i])+","+str(arr_process_time[i])+","
    l += str(arr_Rcut_time[i])+","+str(arr_accRcut[i])+","+str(arr_AMIRcut[i])+","+str(arr_ARIRcut[i])+","
    l += str(arr_PRcut_time[i])+","+str(arr_accPRcut[i])+","+str(arr_AMIPRcut[i])+","+str(arr_ARIPRcut[i])
    f.write(l)
    f.write("\n")
f.close()


"""
Recording an typical solution
-----------------------------
"""


data = pd.read_csv("./Data/MNIST/train.csv")
y = np.array(data['label'])
X = np.array(data.drop(['label'], axis = 1), dtype = np.float64)



list_numbers = [0,1,2,3,6,9]
list_bool = [x in list_numbers for x in y]
yn = y[list_bool]
Xn = X[list_bool,:]

m = np.random.random(Xn.shape[0])
thresh = 0.3

ym = yn[m < thresh]
Xm = Xn[m < thresh,:]


dim = 10
st_time = time.clock()
pca = PCA(n_components=dim)
X_reduced = pca.fit_transform(Xm)
X_graph = kneighbors_graph(X_reduced, n_neighbors=10, include_self = False, mode = 'distance')
n_components, labels = connected_components(X_graph)
print n_components
X_graph.data =  np.exp(-1*(X_graph.data)/((X_graph.data.std()))) + 1e-6
X_graph = 0.5*(X_graph + X_graph.T)
process_time = time.clock() - st_time

# Rcut
st_time = time.clock()
L = laplacian(X_graph)
eigval, embedRcut = eigsh(L, len(list_numbers), sigma = 1e-6)
d0, labelsRcut, d2 = k_means(embedRcut,len(list_numbers))
Rcut_time = time.clock() - st_time

# Calculating the accuracy for Rcut
# Each class obtained on clustering is assumed to belong to the largest intersection
z = confusion_matrix(ym,labelsRcut)
accRcut = 0
for i in range(z.shape[0]):
    accRcut += z[np.argmax(z[:,i]),i]
accRcut = (accRcut+0.0)/len(ym)

# PRcut
st_time = time.clock()
embedPRcut, label_components = PRcut(X_graph, len(list_numbers), discrete_Method='epsilon', epsilon=0.2, verbose = True)
d0, labelsPRcut, d2 = k_means(embedPRcut,len(list_numbers))
PRcut_time = time.clock() - st_time

# Calculating the accuracy for PRcut
# Each class obtained on clustering is assumed to belong to the largest intersection
z = confusion_matrix(ym,labelsPRcut)
accPRcut = 0
for i in range(z.shape[0]):
    accPRcut += z[np.argmax(z[:,i]),i]
accPRcut = (accPRcut+0.0)/len(ym)

print '********************************************'
print '              Dimension = ', dim
print 'time for preprocessing  = ', process_time
print 'Rcut  :      time_taken = ', Rcut_time
print '               Accuracy = ', accRcut
print '                    ARI = ', ARI(ym,labelsRcut)
print '                    AMI = ', AMI(ym,labelsRcut)
print 'PRcut :      time_taken = ', PRcut_time
print '               Accuracy = ', accPRcut
print '                    ARI = ', ARI(ym,labelsPRcut)
print '                    AMI = ', AMI(ym,labelsPRcut)
print '********************************************'


plt.figure()
p = 1
for i in list_numbers:
    plt.subplot(1,len(list_numbers),p)
    plt.axis('off')
    Xtmp = np.mean(Xm[np.where(ym==i),:], axis = 1)
    plt.imshow(Xtmp.reshape((28,28)), cmap = plt.cm.gray)
    p += 1
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('./img/Example6_groundtruth.eps', bbox_inches='tight')

plt.figure()
p = 1
for i in range(len(list_numbers)):
    plt.subplot(1,len(list_numbers),p)
    plt.axis('off')
    Xtmp = np.mean(Xm[np.where(labelsPRcut==i),:], axis = 1)
    plt.imshow(Xtmp.reshape((28,28)), cmap = plt.cm.gray)
    p += 1
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('./img/Example6_PRcut.eps', bbox_inches='tight')

plt.figure()
p = 1
for i in range(len(list_numbers)):
    plt.subplot(1,len(list_numbers),p)
    plt.axis('off')
    Xtmp = np.mean(Xm[np.where(labelsRcut==i),:], axis = 1)
    plt.imshow(Xtmp.reshape((28,28)), cmap = plt.cm.gray)
    p += 1
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('./img/Example6_Rcut.eps', bbox_inches='tight')
plt.close('all')
