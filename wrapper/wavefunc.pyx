import numpy as np
cimport numpy as np

from types cimport cdouble
from grid cimport ShGrid, SpGrid
from sphere_harmonics cimport YlmCache

cdef class SWavefunc:
    def __cinit__(self, ShGrid grid, int m=0, dealloc=True):
        if grid is None:
            self.data = NULL
        else:
            self.data = sh_wavefunc_new(grid.data, m)
        
        self.dealloc = dealloc

    cdef _set_data(self, sh_wavefunc_t* data):
        self.data = data

    def __dealloc__(self):
        if self.dealloc and self.data != NULL:
            sh_wavefunc_del(self.data)

    def n_sp(self, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=2] n = None) -> np.ndarray:
        if n is None:
            n = np.ndarray((grid.data.n[1], grid.data.n[0]), np.double)

        sh_wavefunc_n_sp(self.data, grid.data, &n[0,0], ylm_cache._data)

        return n

    def norm(self):
        return sh_wavefunc_norm(self.data)

    def normalize(self):
        sh_wavefunc_normalize(self.data)

    def z(self):
        return sh_wavefunc_z(self.data)

    def asarray(self):
        cdef cdouble[:, ::1] array = <cdouble[:self.data.grid.n[1],:self.data.grid.n[0]]>self.data.data
        return np.asarray(array)

    def get_sp(self, SpGrid grid, YlmCache ylm_cache, int ir, int ic, int ip):
        return swf_get_sp(self.data, grid.data, [ir, ic, ip], ylm_cache._data)

    @staticmethod
    def random(ShGrid grid, int m=0):
        wf = SWavefunc(grid, m)
        arr = wf.asarray()
        arr[:] = np.random.rand(*arr.shape)
        wf.normalize()
        return wf

cdef SWavefunc swavefunc_from_point(sh_wavefunc_t* data):
    wf = SWavefunc(grid=None, dealloc=False)
    wf._set_data(data)
    return wf
