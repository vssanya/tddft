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

from libc.stdlib cimport malloc, free


cdef class CtWavefunc:
    def __cinit__(self, Sp2Grid grid):
        self.grid = grid
        self.cdata = ct_wavefunc_t(grid.cdata)

    def __init__(self, Sp2Grid grid):
        pass

    def asarray(self):
        cdef complex_t[:, ::1] array = <complex_t[:self.cdata.grid.n[1],:self.cdata.grid.n[0]]>(<complex_t*>self.cdata.data)
        return np.asarray(array)


cdef class SWavefunc:
    def __cinit__(self, ShGrid grid, int m=0, dealloc=True):
        self.grid = grid

        if not dealloc:
            self.cdata = NULL
        else:
            self.cdata = sh_wavefunc_new(grid.data, m)

        self.dealloc = dealloc

    def __init__(self, ShGrid grid, int m=0, dealloc=True):
        pass

    def copy(self):
        cdef SWavefunc wf_copy = SWavefunc(self.grid, m=self.cdata.m)
        sh_wavefunc_copy(self.cdata, wf_copy.cdata)
        return wf_copy

    cdef _set_data(self, sh_wavefunc_t* data):
        self.cdata = data

    def __dealloc__(self):
        if self.dealloc and self.cdata != NULL:
            sh_wavefunc_del(self.cdata)
            self.cdata = NULL

    def n_sp(self, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=2] n = None) -> np.ndarray:
        if n is None:
            n = np.ndarray((grid.data.n[1], grid.data.n[0]), np.double)

        sh_wavefunc_n_sp(self.cdata, grid.data, &n[0,0], ylm_cache.cdata)

        return n

    def norm(self, masked=False):
        if masked:
            return sh_wavefunc_norm(self.cdata, mask_core)
        else:
            return sh_wavefunc_norm(self.cdata, NULL)

    def norm_l(self):
        arr = self.asarray()
        return np.sum(np.abs(arr)**2, axis=1)*self.grid.data.d[0]

    def normalize(self):
        sh_wavefunc_normalize(self.cdata)

    def z(self):
        return sh_wavefunc_z(self.cdata)

    def __mul__(SWavefunc self, SWavefunc other):
        cdef cdouble res = sh_wavefunc_prod(self.cdata, other.cdata)
        return (<complex_t*>(&res))[0]


    def asarray(self):
        cdef complex_t[:, ::1] array = <complex_t[:self.cdata.grid.n[1],:self.cdata.grid.n[0]]>(<complex_t*>self.cdata.data)
        return np.asarray(array)

    def get_sp(self, SpGrid grid, YlmCache ylm_cache, int ir, int ic, int ip):
        cdef cdouble res =  swf_get_sp(self.cdata, grid.data, [ir, ic, ip], ylm_cache.cdata)
        return (<complex_t*>(&res))[0]

    def _figure_data(self, format):
        fig, ax = plt.subplots()
        fig.set_size_inches((6,3))

        ax.plot(self.grid.get_r(), np.sum(np.abs(self.asarray())**2,axis=0))

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
        wf = SWavefunc(grid, m)
        arr = wf.asarray()
        arr[:] = 0.0
        arr[l,:] = np.random.rand(arr.shape[1])
        wf.normalize()
        return wf

    @staticmethod
    def ort_l(wfs, int l):
        cdef sh_wavefunc_t** wf_arr = <sh_wavefunc_t**>malloc(sizeof(sh_wavefunc_t*)*len(wfs))
        for i in range(len(wfs)):
            wf_arr[i] = (<SWavefunc>wfs[i]).cdata

        sh_wavefunc_ort_l(l, len(wfs), wf_arr)
        free(wf_arr)

cdef SWavefunc swavefunc_from_point(sh_wavefunc_t* data, ShGrid grid):
    wf = SWavefunc(grid=grid, dealloc=False)
    wf._set_data(data)
    return wf
