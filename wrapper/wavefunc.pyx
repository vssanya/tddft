import numpy as np
cimport numpy as np

import tdse.utils
if tdse.utils.is_jupyter_notebook():
    import matplotlib.pyplot as plt
    from IPython.core.pylabtools import print_figure

from types cimport complex_t
from abs_pot cimport mask_core
from grid cimport ShGrid, SpGrid
from sphere_harmonics cimport YlmCache

from masks cimport CoreMask

from libc.stdlib cimport malloc, free


cdef class CtWavefunc:
    def __cinit__(self, SpGrid2d grid):
        self.grid = grid
        self.cdata = new cCtWavefunc(<cGrid2d*> grid.cdata)

    def __init__(self, SpGrid2d grid):
        pass

    def asarray(self):
        cdef complex_t[:, ::1] array = <complex_t[:self.cdata.grid.n[1],:self.cdata.grid.n[0]]>(<complex_t*>self.cdata.data)
        return np.asarray(array)

    def norm(self):
        return self.cdata.norm()


cdef class ShWavefunc:
    def __cinit__(self, ShGrid grid, int m=0, dealloc=True):
        self.grid = grid

        if not dealloc:
            self.cdata = NULL
        else:
            self.cdata = new cShWavefunc(grid.data, m)

        self.dealloc = dealloc

    def __init__(self, ShGrid grid, int m=0, dealloc=True):
        pass

    def copy(self):
        cdef ShWavefunc wf_copy = ShWavefunc(self.grid, m=self.cdata.m)
        self.cdata.copy(wf_copy.cdata)
        return wf_copy

    cdef _set_data(self, cShWavefunc* data):
        self.cdata = data

    def __dealloc__(self):
        if self.dealloc and self.cdata != NULL:
            del self.cdata
            self.cdata = NULL

    def n_sp(self, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=2] n = None) -> np.ndarray:
        if n is None:
            n = np.ndarray((grid.data.n[1], grid.data.n[0]), np.double)

        self.cdata.n_sp(grid.data, &n[0,0], ylm_cache.cdata)

        return n

    def norm(self, CoreMask mask = None):
        if mask is None:
            return self.cdata.norm()
        else:
            return self.cdata.norm(<sh_f> mask.cdata[0])

    def norm_l(self):
        arr = self.asarray()
        return np.sum(np.abs(arr)**2, axis=1)*self.grid.data.d[0]

    def normalize(self):
        self.cdata.normalize()

    def z(self, CoreMask mask=None):
        if mask is None:
            return self.cdata.z()
        else:
            return self.cdata.z(<sh_f> mask.cdata[0])

    def pz(self):
        cdef cdouble res = self.cdata.pz()
        return (<complex_t*>(&res))[0]

    def __mul__(ShWavefunc self, ShWavefunc other):
        cdef cdouble res = self.cdata[0]*other.cdata[0]
        return (<complex_t*>(&res))[0]

    def exclude(ShWavefunc self, ShWavefunc other):
        self.cdata.exclude(other.cdata[0])

    def asarray(self):
        cdef complex_t[:, ::1] array = <complex_t[:self.cdata.grid.n[1],:self.cdata.grid.n[0]]>(<complex_t*>self.cdata.data)
        return np.asarray(array)

    def get_sp(self, SpGrid grid, YlmCache ylm_cache, int ir, int ic, int ip):
        cdef cdouble res =  self.cdata.get_sp(grid.data, [ir, ic, ip], ylm_cache.cdata)
        return (<complex_t*>(&res))[0]

    def _figure_data(self, format):
        fig, ax = plt.subplots()
        fig.set_size_inches((6,3))

        ax.plot(self.grid.r, np.sum(np.abs(self.asarray())**2,axis=0))

        ax.set_xlabel('r, (a.u.)')
        ax.set_ylabel(r'$\left|\psi\right|^2$, (a.u.)')

        ax.set_yscale('log')

        data = print_figure(fig, format)
        plt.close(fig)
        return data

    def _repr_png_(self):
        return self._figure_data('png')

    @staticmethod
    def random(ShGrid grid, int l=0, int m=0):
        wf = ShWavefunc(grid, m)
        arr = wf.asarray()
        arr[:] = 0.0
        arr[l,:] = np.random.rand(arr.shape[1])
        wf.normalize()
        return wf

    @staticmethod
    def ort_l(wfs, int l):
        cdef cShWavefunc** wf_arr = <cShWavefunc**>malloc(sizeof(cShWavefunc*)*len(wfs))
        for i in range(len(wfs)):
            wf_arr[i] = (<ShWavefunc>wfs[i]).cdata

        cShWavefunc.ort_l(l, len(wfs), wf_arr)
        free(wf_arr)

cdef ShWavefunc swavefunc_from_point(cShWavefunc* data, ShGrid grid, bint dealloc):
    wf = ShWavefunc(grid=grid, dealloc=dealloc)
    wf._set_data(data)
    return wf


cdef class ShNeWavefunc:
    def __cinit__(self, ShNeGrid grid, int m=0, dealloc=True):
        self.grid = grid

        if not dealloc:
            self.cdata = NULL
        else:
            self.cdata = new cShNeWavefunc(grid.data, m)

        self.dealloc = dealloc

    def __init__(self, ShNeGrid grid, int m=0, dealloc=True):
        pass

    def copy(self):
        cdef ShNeWavefunc wf_copy = ShNeWavefunc(self.grid, m=self.cdata.m)
        self.cdata.copy(wf_copy.cdata)
        return wf_copy

    cdef _set_data(self, cShNeWavefunc* data):
        self.cdata = data

    def __dealloc__(self):
        if self.dealloc and self.cdata != NULL:
            del self.cdata
            self.cdata = NULL

    def n_sp(self, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=2] n = None) -> np.ndarray:
        if n is None:
            n = np.ndarray((grid.data.n[1], grid.data.n[0]), np.double)

        self.cdata.n_sp(grid.data, &n[0,0], ylm_cache.cdata)

        return n

    def norm(self, CoreMask mask = None):
        if mask is None:
            return self.cdata.norm()
        else:
            return self.cdata.norm(<sh_f> mask.cdata[0])

    def norm_l(self):
        arr = self.asarray()
        return np.sum(np.abs(arr)**2, axis=1)*self.grid.data.d[0]

    def normalize(self):
        self.cdata.normalize()

    def z(self, CoreMask mask=None):
        if mask is None:
            return self.cdata.z()
        else:
            return self.cdata.z(<sh_f> mask.cdata[0])

    def pz(self):
        cdef cdouble res = self.cdata.pz()
        return (<complex_t*>(&res))[0]

    def __mul__(ShNeWavefunc self, ShNeWavefunc other):
        cdef cdouble res = self.cdata[0]*other.cdata[0]
        return (<complex_t*>(&res))[0]

    def exclude(ShNeWavefunc self, ShNeWavefunc other):
        self.cdata.exclude(other.cdata[0])

    def asarray(self):
        cdef complex_t[:, ::1] array = <complex_t[:self.cdata.grid.n[1],:self.cdata.grid.n[0]]>(<complex_t*>self.cdata.data)
        return np.asarray(array)

    def get_sp(self, SpGrid grid, YlmCache ylm_cache, int ir, int ic, int ip):
        cdef cdouble res =  self.cdata.get_sp(grid.data, [ir, ic, ip], ylm_cache.cdata)
        return (<complex_t*>(&res))[0]

    def _figure_data(self, format):
        fig, ax = plt.subplots()
        fig.set_size_inches((6,3))

        ax.plot(self.grid.r, np.sum(np.abs(self.asarray())**2,axis=0))

        ax.set_xlabel('r, (a.u.)')
        ax.set_ylabel(r'$\left|\psi\right|^2$, (a.u.)')

        ax.set_yscale('log')

        data = print_figure(fig, format)
        plt.close(fig)
        return data

    def _repr_png_(self):
        return self._figure_data('png')

    @staticmethod
    def random(ShNeGrid grid, int l=0, int m=0):
        wf = ShNeWavefunc(grid, m)
        arr = wf.asarray()
        arr[:] = 0.0
        arr[l,:] = np.random.rand(arr.shape[1])
        wf.normalize()
        return wf

    @staticmethod
    def ort_l(wfs, int l):
        cdef cShNeWavefunc** wf_arr = <cShNeWavefunc**>malloc(sizeof(cShNeWavefunc*)*len(wfs))
        for i in range(len(wfs)):
            wf_arr[i] = (<ShNeWavefunc>wfs[i]).cdata

        cShNeWavefunc.ort_l(l, len(wfs), wf_arr)
        free(wf_arr)
