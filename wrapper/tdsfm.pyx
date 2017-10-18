from cpython cimport Py_buffer

import numpy as np
cimport numpy as np

from types cimport cdouble
from grid cimport ShGrid, SpGrid
from field cimport Field
from wavefunc cimport SWavefunc


cdef class TDSFM:
    def __cinit__(self, SpGrid k_grid, ShGrid r_grid, int ir):
        self.cdata = tdsfm_new(k_grid.data, r_grid.data, ir)
        self.k_grid = k_grid
        self.r_grid = r_grid

    def __init__(self, SpGrid k_grid, ShGrid r_grid, int ir):
        pass

    def __dealloc__(self):
        tdsfm_del(self.cdata)

    def calc(self, Field field, SWavefunc wf, double t, double dt):
        tdsfm_calc(self.cdata, field.cdata, wf.cdata, t, dt)

    def calc_inner(self, Field field, SWavefunc wf, double t):
        tdsfm_calc_inner(self.cdata, field.cdata, wf.cdata, t)

    def asarray(self):
        cdef cdouble[:, ::1] arr = <cdouble[:self.cdata.k_grid.n[1], :self.cdata.k_grid.n[0]]>self.cdata.data
        return np.asarray(arr)
