import numpy as np

from types cimport cdouble
from grid cimport SGrid, SpGrid

cdef class SWavefunc:
    def __cinit__(self, SGrid grid, int m=0):
        self.data = sphere_wavefunc_new(grid.data, m)

    def __dealloc__(self):
        sphere_wavefunc_del(self.data)

    def norm(self):
        sphere_wavefunc_norm(self.data)

    def normalize(self):
        sphere_wavefunc_normalize(self.data)

    def z(self):
        sphere_wavefunc_z(self.data)

    def asarray(self):
        cdef cdouble[:, ::1] array = <cdouble[:self.data.grid.n[0],:self.data.grid.n[1]]>self.data.data
        return np.asarray(array)

    def get_sp(self, SpGrid grid, int ir, int ic, int ip):
        return swf_get_sp(self.data, grid.data, [ir, ic, ip])
