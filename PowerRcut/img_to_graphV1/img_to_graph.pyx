#cython: profile = True

cimport numpy as cnp
import numpy as np
from cython cimport boundscheck, wraparound

from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph as knn


def method1(int[:,:,:] im, int s0, int s1, int s2,int spat_distance = 1, double beta = 1., double eps = 1e-5 ):
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
                        if x2 >= 0 and x2 < s0 and y2 >= 0 and y2 < s1 and (x1 != x2 or y1 != y2):
                            if abs(x1-x2) + abs(y1-y2) <= spat_distance:
                                ind1 = x1*s1 + y1
                                ind2 = x2*s1 + y2
                                w1 = 0.
                                w3 = 1000
                                for i in range(s2):
                                    w1 += (<float> np.abs(im[x1,y1,i] - im[x2,y2,i]))/3*255.
                                    if w3 > (255 - np.abs(im[x1,y1,i] - im[x2,y2,i])):
                                        w3 = 255 - np.abs(im[x1,y1,i] - im[x2,y2,i])
                                w2 = <float> np.sqrt((x1-x2)**2 + (y1 - y2)**2)
                                mv_ind1[ind] = ind1
                                mv_ind2[ind] = ind2
                                mv_wInt[ind] = w1
                                mv_wSpat[ind] = w2
                                mv_wBucket[ind] = w3
                                ind += 1



    list_1 = np.asarray(mv_ind1[:ind])
    list_2 = np.asarray(mv_ind2[:ind])

    weight_int = np.asarray(mv_wInt[:ind])
    weight_spat = np.asarray(mv_wSpat[:ind])
    weight_bucket = np.asarray(mv_wBucket[:ind])

    graph_bucket = csr_matrix((weight_bucket, (list_1, list_2)), shape = (s0*s1, s0*s1))

    if spat_distance > 1:
        graph_int = csr_matrix((weight_int, (list_1, list_2)), shape = (s0*s1, s0*s1))
        graph_spat = csr_matrix((weight_spat, (list_1, list_2)), shape = (s0*s1, s0*s1))
        graph_int.data = np.exp(-beta*(graph_int.data**2)/(graph_int.data.std()**2)) + eps
        graph_spat.data = np.exp(-beta*(graph_spat.data**2)/(graph_spat.data.std()**2)) + eps
        graph = graph_int.multiply(graph_spat)
        return graph, graph_bucket
    else:
        graph_int = csr_matrix((weight_int, (list_1, list_2)), shape = (s0*s1, s0*s1))
        graph_int.data = np.exp(-beta*(graph_int.data)/(graph_int.data.std())) + eps
        return graph_int, weight_int



def method1_hyperspectral(int[:,:,:] im, int s0, int s1, int s2,int spat_distance = 1, double beta = 1., double eps = 1e-5 ):
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
    cdef float w1, w2, c0, c1
    with boundscheck(True), wraparound(True):
        for x1 in range(s0):
            for y1 in range(s1):
                for x2 in range(x1-spat_distance, x1+spat_distance+1):
                    for y2 in range(y1-spat_distance, y1+spat_distance+1):
                        if x2 >= 0 and x2 < s0 and y2 >= 0 and y2 < s1 and (x1 != x2 or y1 != y2):
                            if abs(x1-x2) + abs(y1-y2) <= spat_distance:
                                ind1 = x1*s1 + y1
                                ind2 = x2*s1 + y2
                                w1 = 0.
                                w3 = 1000
                                c0 = 0.
                                c1 = 0.
                                denom = 0.
                                for i in range(s2):
                                    w1 += (<float> np.abs(im[x1,y1,i] - im[x2,y2,i]))
                                w2 = <float> np.sqrt((x1-x2)**2 + (y1 - y2)**2)
                                mv_ind1[ind] = ind1
                                mv_ind2[ind] = ind2
                                mv_wInt[ind] = w1
                                mv_wSpat[ind] = w2
                                mv_wBucket[ind] = w3
                                ind += 1



    list_1 = np.asarray(mv_ind1[:ind])
    list_2 = np.asarray(mv_ind2[:ind])

    weight_int = np.asarray(mv_wInt[:ind])
    weight_spat = np.asarray(mv_wSpat[:ind])
    weight_bucket = np.asarray(mv_wBucket[:ind])

    graph_bucket = csr_matrix((weight_bucket, (list_1, list_2)), shape = (s0*s1, s0*s1))

    if spat_distance > 1:
        graph_int = csr_matrix((weight_int, (list_1, list_2)), shape = (s0*s1, s0*s1))
        graph_spat = csr_matrix((weight_spat, (list_1, list_2)), shape = (s0*s1, s0*s1))
        graph_int.data = np.exp(-beta*(graph_int.data**2)/(graph_int.data.std()**2)) + eps
        graph_spat.data = np.exp(-beta*(graph_spat.data**2)/(graph_spat.data.std()**2)) + eps
        graph = graph_int.multiply(graph_spat)
        return graph, graph_bucket
    else:
        graph_int = csr_matrix((weight_int, (list_1, list_2)), shape = (s0*s1, s0*s1))
        graph_int.data = np.exp(-beta*(graph_int.data)/(graph_int.data.std())) + eps
        # graph_int.data = weight_int
        return graph_int, weight_int
