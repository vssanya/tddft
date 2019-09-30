import numpy as np
cimport numpy as np

from grid cimport SpGrid
from array cimport ArraySp2D

cdef class JlCache:
    def __cinit__(self, SpGrid grid, int l_max):
        self.cdata = new cJlCache(grid.data, l_max)

    def __init__(self, SpGrid grid, int l_max):
        pass

    def __dealloc__(self):
        del self.cdata

    def __call__(self, double r, int l):
        return self.cdata[0](r, l)

    @staticmethod
    def calc(double r, int l):
        return cJlCache.calc(r, l)

cdef class YlmCache:
    def __cinit__(self, int l_max, SpGrid grid):
        self.cdata = new cYlmCache(grid.data, l_max)

    def __init__(self, int l_max, SpGrid grid):
        pass

    def __dealloc__(self):
        if self.cdata != NULL:
            del self.cdata

    def get(self, int l, int m, int ic):
        return self.cdata[0](l, m, ic)

    @np.vectorize
    def calc(self, int l, int m, double c):
        return self.cdata[0](l, m, c)

    def __call__(self, int l, int m, double c):
        return self.cdata[0](l, m, c)

def series(np.ndarray[np.double_t, ndim=2, mode='c'] arr, int l, int m, SpGrid grid, YlmCache cache):
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] res = np.ndarray(grid.data.n[0], dtype=np.double)
    cdef ArraySp2D[double]* arr_c = new ArraySp2D[double](&arr[0, 0], grid.data.getGrid2d())
    sh_series(arr_c, l, m, <double*>&res[0], cache.cdata)
    del arr_c
    return res

@np.vectorize
def sh_clebsch_gordan_coef(int j1, int m1, int j2, int m2, int J, int M):
    return clebsch_gordan_coef(j1, m1, j2, m2, J, M)

@np.vectorize
def sh_y3(int l1, int m1, int l2, int m2, int L, int M):
    return y3(l1, m1, l2, m2, L, M)
