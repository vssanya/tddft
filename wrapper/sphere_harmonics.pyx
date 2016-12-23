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
def ylm(int l, int m, double x):
    return Ylm(l, m, x)
