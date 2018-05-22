import numpy as np
cimport numpy as np

from grid cimport SpGrid

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

cdef object func_2d = None

cdef double func_2d_wrapper(int i1, int i2):
    return func_2d(i1, i2)

def series(func, int l, int m, SpGrid grid, YlmCache cache):
    global func_2d

    func_2d = func
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] res = np.ndarray(grid.data.n[0], dtype=np.double)
    sh_series(func_2d_wrapper, l, m, grid.data, &res[0], cache.cdata)
    return res

@np.vectorize
def sh_clebsch_gordan_coef(int j1, int m1, int j2, int m2, int J, int M):
    return clebsch_gordan_coef(j1, m1, j2, m2, J, M)

@np.vectorize
def sh_y3(int l1, int m1, int l2, int m2, int L, int M):
    return y3(l1, m1, l2, m2, L, M)
