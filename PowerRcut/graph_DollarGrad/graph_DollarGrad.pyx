#cython: profile = True

cimport numpy as np
import numpy as np
from cython cimport boundscheck, wraparound

from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph as knn


def get_affinityDollatGrad(img, spat_distance = 1,  beta = 10., eps = 1e-5 ):
    s0, s1 = img.shape
    return method1(img, s0, s1, spat_distance = spat_distance,  beta = beta, eps = eps )

def method1(int[:,:] im, int s0, int s1, int spat_distance = 1, double beta = 1., double eps = 1e-5 ):
    """
    method1(im, s0, s1, s2, spat_distance, beta, eps)

    Create a similarity graph from the image.

    Parameters
    -----------
    im : Image to be converted to graph. Should be int32.
    s0, s1 : Spatial dimensions of the image.
    s2 : Number of bands
    spat_distance : The distance to which the edges must be added. Default: 1 for 4-adjacency.
    beta, eps : Parameters used for taking similarity. Default : (1., 1e-5) [exp(-beta*data/data.stdev()) + eps]


    Returns
    -------
    if spat_distance > 1 :
        graph with edges weighted by product of intensity and spatial distance
        AND graph with edges weighted by 255 - max(intensity difference of bands)
    if spat_distance = 1:
        graph with edges weighted by intensity distance
        AND graph with edges weighted by 255 - max(intensity difference of bands)

    """
    cdef int size = s0*s1*(spat_distance+1)*(spat_distance+1)*2
    cdef int[:] mv_ind1 = np.zeros(size, dtype = np.int32), mv_ind2 = np.zeros(size, dtype = np.int32)
    cdef double[:] mv_wSpat = np.zeros(size, dtype = np.float64), mv_wInt = np.zeros(size, dtype = np.float64)
    cdef int[:] mv_wBucket = np.zeros(size, dtype = np.int32)
    cdef int x1 = 0, x2 = 0, y1 = 0, y2 = 0, ind1, ind2, i, ind = 0, w3
    cdef float w1, w2
    with boundscheck(True), wraparound(True):
        for x1 in range(s0):
            for y1 in range(s1):
                for x2 in range(x1-spat_distance, x1+spat_distance+1):
                    for y2 in range(y1-spat_distance, y1+spat_distance+1):
                        if (x2 >= 0) and (x2 <s0) and (y2 >= 0) and (y2 < s1) and ((x1 != x2) or (y1 != y2)):
                            if abs(x1-x2) + abs(y1-y2) <= spat_distance:
                                ind1 = x1*s1 + y1
                                ind2 = x2*s1 + y2
                                w1 = (<float> (im[x1,y1] + im[x2,y2]))/2*255.
                                # w1 = ((<float> im[x1,y1])/255)*((<float> im[x2,y2])/255)
                                w3 = 0
                                w2 = <float> np.sqrt((x1-x2)**2 + (y1 - y2)**2)
                                mv_ind1[ind] = ind1
                                mv_ind2[ind] = ind2
                                mv_wInt[ind] = w1
                                mv_wSpat[ind] = w2
                                mv_wBucket[ind] = w3
                                ind += 1


    graph_bucket = csr_matrix((mv_wBucket[:ind], (mv_ind1[:ind], mv_ind2[:ind])), shape = (s0*s1, s0*s1))

    if spat_distance > 1:
        graph_int = csr_matrix((mv_wInt[:ind], (mv_ind1[:ind], mv_ind2[:ind])), shape = (s0*s1, s0*s1))
        graph_spat = csr_matrix((mv_wSpat[:ind], (mv_ind1[:ind], mv_ind2[:ind])), shape = (s0*s1, s0*s1))
        graph_int.data = np.exp(-beta*(graph_int.data**2)/(graph_int.data.std()**2)) + eps
        graph_spat.data = np.exp(-beta*(graph_spat.data**2)/(graph_spat.data.std()**2)) + eps
        graph = graph_int.multiply(graph_spat)
        return None, graph_bucket
    else:
        graph_int = csr_matrix((mv_wInt[:ind], (mv_ind1[:ind], mv_ind2[:ind])), shape = (s0*s1, s0*s1))
        graph_int.data = np.exp(-beta*(graph_int.data**2)/(graph_int.data.std()**2)) + eps
        return graph_int, graph_bucket
