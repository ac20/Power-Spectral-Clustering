import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from collections import Counter
import time

from sklearn.cluster import k_means

def _kmeans_discrete(weights, nClust):
    """
    """
    c0, c1, c2 = k_means(weights.reshape((weights.shape[0],1)), nClust, n_init=1)
    w_thresh = []
    for i in range(nClust):
        w_thresh.append(np.max(weights[np.where(c1==i)]))
    return np.sort(w_thresh)

def _epsilon_discrete(weights, epsilon):
    """
    """
    return np.sort(np.unique(np.round(weights/epsilon)))*epsilon

def discretize(weights, method, nClust, epsilon):
    """Buckets the weights for processing


    """
    if method == 'kmeans':
        return _kmeans_discrete(weights, nClust)
    elif method == 'epsilon':
        return _epsilon_discrete(weights, epsilon)
    else:
        raise Exception("method can only be 'kmeans' or 'epsilon'.")

# @profile
def PRcut(graph, k = 6, discrete_Method = 'kmeans', epsilon = 0.01, nClust = 3, threshComp=0, verbose = False):
    """ Returns the Power Ratio embedding of the graph

    Parameters
    ----------
    graph: sparse matrix (n_dataPoints, n_dataPoints)
        adjacency matrix of the similarity graph in sparse format
    k: int
        Number of eigenvectors required
    discrete_Method: 'kmeans' or 'epsilon'
        Method used to discretize the weights. Should be either 'kmeans' or 'epsilon'
    epsilon: float
        parameter used for epsilon bucketing. Ignored otherwise.
    nClust: int
        parameter used for kmeans bucketing. Ignored otherwise.
    verbose: bool
        Prints various values throughout the execution.

    Returns
    ---------
    N: Power Ratio cut embedding
    label_components: connected componets after MST phase

    """

    weights = np.array(graph.data, copy = True)
    indices = np.array(graph.indices, copy = True)
    indptr =  np.array(graph.indptr, copy = True)

    # Degenerate case of Ratio cut
    if nClust == 1:
        L = laplacian(graph)
        eigval, embedNcut = eigsh(L, k=20, sigma = 1e-6 )
        return embedNcut, eigval

    # if nClust > 1
    sizeGraph = graph.shape[0]
    G = graph.copy()
    G_tmp = graph.copy()

    weights_thresh = discretize(weights, method=discrete_Method, nClust=nClust, epsilon=epsilon)
    weightsZero = np.ones(weights.shape[0], dtype = np.float64)
    weightsOne =  np.zeros(weights.shape[0], dtype = np.float64)
    for i in range(len(weights_thresh)):
        weightsZero[weights < weights_thresh[i]] = 0
        weightsOne[weights < weights_thresh[i]] = 1

        G_tmp = csr_matrix((np.array(weightsZero), np.array(indices), np.array(indptr)), shape = graph.shape)
        G_tmp.eliminate_zeros()
        n_comp = connected_components(G_tmp, directed = False, return_labels = False)
        if verbose:
            print n_comp, weights_thresh[i]
        if n_comp >= k:
            break

    n_components, label_components = connected_components(G_tmp,directed = False)

    if threshComp > 0:
        count_labels = Counter(label_components)
        label_componentsNew = filter(lambda x: count_labels[x] > threshComp, label_components)
        n_componentsNew = len(np.unique(label_componentsNew))
        d = dict(zip(np.unique(label_componentsNew),np.arange(n_componentsNew)))
        label_componentsNew = [d[x] for x in label_componentsNew]
        vertices = np.arange(sizeGraph)
        verticesNew = filter(lambda x: count_labels[label_components[x]] > threshComp, list(vertices))
    else:
        label_componentsNew, n_componentsNew, verticesNew = label_components, n_components, np.arange(sizeGraph)

    u = np.array(verticesNew, dtype = np.int32)
    v = np.array(label_componentsNew, dtype = np.int32)
    z = Counter(v)
    w = 1./np.sqrt(np.array([z[i] for i in v], dtype = np.float64))

    N = csr_matrix((w,(u,v)),shape=(sizeGraph,n_componentsNew))

    G = csr_matrix((np.array(weights*weightsOne), np.array(indices), np.array(indptr)), shape = graph.shape)
    G.eliminate_zeros()
    L = laplacian(G, normed=True)

    Lp = (N.T).dot(L.dot(N))
    if verbose:
        print 'size of the eigenvalue problem (Before) : ', L.shape
        print 'size of the eigenvalue problem  (After) : ', Lp.shape
    if Lp.shape[0] > k+1:
        eigval, eigvec = eigsh(Lp, k, sigma = 1e-6)
    else:
        eigval, eigvec = eigh(Lp.toarray())
    return N.dot(eigvec[:,:k]), label_components
