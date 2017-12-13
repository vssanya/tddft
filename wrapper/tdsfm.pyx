from cpython cimport Py_buffer

import numpy as np
cimport numpy as np

from types cimport complex_t
from grid cimport ShGrid, SpGrid, sp_grid_r, sp_grid_c
from field cimport Field
from wavefunc cimport SWavefunc
from sphere_harmonics cimport ylm_cache_get


cdef class TDSFM:
    def __cinit__(self, SpGrid k_grid, ShGrid r_grid, double A_max, int ir, bool init_cache = True):
        self.cdata = new tdsfm_t(k_grid.data, r_grid.data, A_max, ir, init_cache)
        self.k_grid = k_grid
        self.r_grid = r_grid

    def __init__(self, SpGrid k_grid, ShGrid r_grid, double A_max, int ir, bool init_cache = True):
        pass

    def __dealloc__(self):
        del self.cdata

    def pz(self):
        return self.cdata.pz()

    def init_cache(self):
        self.cdata.init_cache()

    def calc(self, Field field, SWavefunc wf, double t, double dt):
        self.cdata[0].calc(field.cdata, wf.cdata[0], t, dt)

    def calc_inner(self, Field field, SWavefunc wf, double t, int ir_min = 0, int ir_max = -1):
        if ir_max == -1:
            ir_max = self.cdata.ir
        self.cdata[0].calc_inner(field.cdata, wf.cdata[0], t, ir_min, ir_max)

    def asarray(self):
        cdef complex_t[:, ::1] arr = <complex_t[:self.cdata.k_grid.n[1], :self.cdata.k_grid.n[0]]>(<complex_t*>self.cdata.data)
        return np.asarray(arr)
