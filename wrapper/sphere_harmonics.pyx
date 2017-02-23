import numpy as np
cimport numpy as np

from grid cimport SpGrid

cdef object func_2d = None

cdef double func_2d_wrapper(int i1, int i2):
    return func_2d(i1, i2)

def series(func, int l, int m, SpGrid grid):
    global func_2d

    func_2d = func
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] res = np.ndarray(grid.data.n[0], dtype=np.double)
    sh_series(func_2d_wrapper, l, m, grid.data, &res[0])
    return res

@np.vectorize
def sp_ylm(int l, int m, int c):
    return ylm(l, m, c)

@np.vectorize
def sh_clebsch_gordan_coef(int j1, int m1, int j2, int m2, int J, int M):
    return clebsch_gordan_coef(j1, m1, j2, m2, J, M)

@np.vectorize
def sh_y3(int l1, int m1, int l2, int m2, int L, int M):
    return y3(l1, m1, l2, m2, L, M)
