import numpy as np
cimport numpy as np

cdef class DoubleArray2D:
    def __cinit__(self, np.ndarray[np.double_t, ndim=2] data):
        self.cdata = new Array2D[double](&data[0,0], cGrid2d(data.shape[1], data.shape[0]))

    def __dealloc__(self):
        del self.cdata

    def __init__(self, np.ndarray[np.double_t, ndim=2] data):
        pass
