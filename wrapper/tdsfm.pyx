from cpython cimport Py_buffer

import numpy as np
cimport numpy as np

import scipy.special

from types cimport cdouble
from grid cimport ShGrid, SpGrid, sp_grid_r, sp_grid_c
from field cimport Field
from wavefunc cimport SWavefunc
from sphere_harmonics cimport ylm_cache_get


def sph_jn(int l, double x):
    return jn(l, x)


cdef class TDSFM:
    def __cinit__(self, SpGrid k_grid, ShGrid r_grid, double A_max, int ir):
        self.cdata = tdsfm_new(k_grid.data, r_grid.data, A_max, ir)
        self.k_grid = k_grid
        self.r_grid = r_grid

    def __init__(self, SpGrid k_grid, ShGrid r_grid, int ir):
        pass

    def __dealloc__(self):
        tdsfm_del(self.cdata)

    def calc(self, Field field, SWavefunc wf, double t, double dt):
        tdsfm_calc(self.cdata, field.cdata, wf.cdata, t, dt)

    def calc_inner(self, Field field, SWavefunc wf, double t, int ir_min = 0, int ir_max = -1):
        if ir_max == -1:
            ir_max = self.cdata.ir
        tdsfm_calc_inner(self.cdata, field.cdata, wf.cdata, t, ir_min, ir_max)

    def asarray(self):
        cdef cdouble[:, ::1] arr = <cdouble[:self.cdata.k_grid.n[1], :self.cdata.k_grid.n[0]]>self.cdata.data
        return np.asarray(arr)
