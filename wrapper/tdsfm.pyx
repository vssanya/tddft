from cpython cimport Py_buffer

import numpy as np
cimport numpy as np

from types cimport complex_t
from grid cimport ShGrid, SpGrid
from field cimport Field
from wavefunc cimport ShWavefunc
from orbitals cimport ShOrbitals
from workspace cimport LENGTH, VELOCITY


cdef class TDSFM:
    def __cinit__(self, SpGrid k_grid, ShGrid r_grid, int ir, bool init_cache = True):
        pass

    def __dealloc__(self):
        del self.cdata

    def pz(self):
        return self.cdata.pz()

    def norm(self):
        return self.cdata.norm()

    def init_cache(self):
        self.cdata.init_cache()

    def calc(self, Field field, ShWavefunc wf, double t, double dt, double mask = 1.0):
        self.cdata[0].calc(field.cdata, wf.cdata[0], t, dt, mask)

    def calc_inner(self, Field field, ShWavefunc wf, double t, int ir_min = 0, int ir_max = -1, int l_min = 0, int l_max = -1):
        if ir_max == -1:
            ir_max = self.cdata.ir
        self.cdata[0].calc_inner(field.cdata, wf.cdata[0], t, ir_min, ir_max, l_min, l_max)

    def calc_norm_k(self, ShWavefunc wf, int ir_min = 0, int ir_max = -1, int l_min = 0, int l_max = -1):
        if ir_max == -1:
            ir_max = wf.grid.Nr

        self.cdata[0].calc_norm_k(wf.cdata[0], ir_min, ir_max, l_min, l_max)

    def asarray(self):
        cdef complex_t[:, ::1] arr = <complex_t[:self.cdata.k_grid.n[1], :self.cdata.k_grid.n[0]]>(<complex_t*>self.cdata.data)
        return np.asarray(arr)

cdef class TDSFM_LG(TDSFM):
    def __cinit__(self, SpGrid k_grid, ShGrid r_grid, int ir, bool init_cache = True, int m_max=2, double A_max = 0.0):
        self.cdata = <TDSFM_Base*> new TDSFM_E(k_grid.data[0], r_grid.data[0], A_max, ir, m_max, init_cache)
        self.k_grid = k_grid
        self.r_grid = r_grid

    def __init__(self, SpGrid k_grid, ShGrid r_grid, int ir, bool init_cache = True, int m_max=2, double A_max = 0.0):
        pass

cdef class TDSFM_VG(TDSFM):
    def __cinit__(self, SpGrid k_grid, ShGrid r_grid, int ir, int m_max=2, bool init_cache = True):
        self.cdata = <TDSFM_Base*> new TDSFM_A(k_grid.data[0], r_grid.data[0], ir, m_max, init_cache)
        self.k_grid = k_grid
        self.r_grid = r_grid

    def __init__(self, SpGrid k_grid, ShGrid r_grid, int ir, int m_max=2, bool init_cache = True):
        pass

cdef class TDSFMOrbs:
    def __cinit__(self, ShOrbitals orbs, SpGrid k_grid, int ir, int gauge = 0, bool init_cache = True, double A_max = 0.0):
        cdef Gauge c_gauge
        if gauge == 0:
            c_gauge = LENGTH
        else:
            c_gauge = VELOCITY

        self.cdata = new cTDSFMOrbs(orbs.cdata[0], k_grid.data[0], ir, c_gauge, init_cache, A_max)

    def __init__(self, ShOrbitals orbs, SpGrid k_grid, int ir, int gauge = 0, bool init_cache = True, double A_max = 0.0):
        pass

    def calc(self, Field field, ShOrbitals orbs, double t, double dt, double mask = 1.0):
        self.cdata[0].calc(field.cdata, orbs.cdata[0], t, dt, mask)

    def calc_inner(self, Field field, ShOrbitals orbs, double t, int ir_min = 0, int ir_max = -1, int l_min = 0, int l_max = -1):
        if ir_max == -1:
            ir_max = self.cdata.tdsfmWf.ir
        self.cdata[0].calc_inner(field.cdata, orbs.cdata[0], t, ir_min, ir_max, l_min, l_max)

    def collect(self, np.ndarray[complex_t, ndim=3] dest):
        cdef cdouble* res_ptr = NULL
        if self.cdata.pOrbs.is_root():
            res_ptr = <cdouble*>dest.data

        self.cdata.collect(res_ptr)
