import numpy as np

from types cimport cdouble
from grid cimport SGrid, SpGrid

cdef class SWavefunc:
    def __cinit__(self, SGrid grid, int m=0, dealloc=True):
        if grid is None:
            self.data = NULL
        else:
            self.data = sphere_wavefunc_new(grid.data, m)
        
        self.dealloc = dealloc

    cdef _set_data(self, sphere_wavefunc_t* data):
        self.data = data

    def __dealloc__(self):
        if self.dealloc and self.data != NULL:
            sphere_wavefunc_del(self.data)

    def norm(self):
        return sphere_wavefunc_norm(self.data)

    def normalize(self):
        sphere_wavefunc_normalize(self.data)

    def z(self):
        return sphere_wavefunc_z(self.data)

    def asarray(self):
        cdef cdouble[:, ::1] array = <cdouble[:self.data.grid.n[1],:self.data.grid.n[0]]>self.data.data
        return np.asarray(array)

    def get_sp(self, SpGrid grid, int ir, int ic, int ip):
        return swf_get_sp(self.data, grid.data, [ir, ic, ip])

    @staticmethod
    def random(SGrid grid, int m=0):
        wf = SWavefunc(grid, m)
        arr = wf.asarray()
        arr[:] = np.random.rand(*arr.shape)
        wf.normalize()
        return wf

cdef SWavefunc swavefunc_from_point(sphere_wavefunc_t* data):
    wf = SWavefunc(grid=None, dealloc=False)
    wf._set_data(data)
    return wf
