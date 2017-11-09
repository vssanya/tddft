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

    def calc_inner(self, Field field, SWavefunc wf, double t, int ir_min = 0, int ir_max = -1):
        if ir_max == -1:
            ir_max = self.cdata.ir
        tdsfm_calc_inner(self.cdata, field.cdata, wf.cdata, t, ir_min, ir_max)

    def asarray(self):
        cdef cdouble[:, ::1] arr = <cdouble[:self.cdata.k_grid.n[1], :self.cdata.k_grid.n[0]]>self.cdata.data
        return np.asarray(arr)


cdef class TDSFM_new:
    def __cinit__(self, SpGrid k_grid, ShGrid r_grid, int ir):
        self.k_grid = k_grid
        self.r_grid = r_grid
        self.data = np.zeros((k_grid.data.n[1], k_grid.data.n[0]), dtype=np.complex)
        self.jl = np.ndarray((k_grid.data.n[0], r_grid.data.n[1]+1), dtype=np.double)
        self.ylm = YlmCache(r_grid.data.n[1], k_grid)

        cdef double r = r_grid.data.d[0]*ir
        r_k = k_grid.get_r()*r
        cdef int il
        cdef int ik
        for ik in range(self.jl.shape[0]):
            for il in range(self.jl.shape[1]):
                self.jl[ik,il] = scipy.special.spherical_jn(il, r_k[ik])

        self.cdata.k_grid = k_grid.data
        self.cdata.r_grid = r_grid.data
        self.cdata.ir = ir
        self.cdata.data = &self.data[0,0]
        self.cdata.jl = &self.jl[0,0]
        self.cdata.int_A = 0
        self.cdata.int_A2 = 0

    def asarray(self):
        return np.asarray(self.data)

    def __init__(self, SpGrid k_grid, ShGrid r_grid, int ir):
        pass

    def calc(self, Field field, SWavefunc wf, double t, double dt):
        tdsfm_calc(&self.cdata, field.cdata, wf.cdata, t, dt)

    def calc_inner(self, Field field, SWavefunc wf, double t, int ir_min = 0, int ir_max = -1):
        cdef int ik
        cdef int ic
        cdef int il
        cdef int ir
        cdef double k
        cdef double kz
        cdef cdouble a_k

        psi = wf.asarray()

        for ik in range(self.k_grid.data.n[0]):
            for ic in range(self.k_grid.data.n[1]):
                k = sp_grid_r(self.k_grid.data, ik)
                kz = sp_grid_c(self.k_grid.data, ic)*k

                a_k = 0.0
                for il in range(wf.cdata.grid.n[1]):
                    r = self.r_grid.get_r()
                    a_k += np.sum(r*psi[il]*scipy.special.spherical_jn(il, k*r))*(-1.0j)**il*ylm_cache_get(self.ylm.cdata, il, wf.cdata.m, ic);

                self.data[ic, ik] += a_k*np.sqrt(2.0/np.pi)*self.r_grid.data.d[0];
