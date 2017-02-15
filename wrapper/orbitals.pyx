import numpy as np

from types cimport cdouble
from grid cimport SGrid, SpGrid

cdef class SOrbitals:
    def __cinit__(self, int ne, SGrid grid):
        self._data = ks_orbials_new(ne, grid.data)

    def __dealloc__(self):
        ks_orbitals_del(self._data)
    
    def n(self, SpGrid grid, int ir, int ic):
        return ks_orbitals_n(self._data, grid.data, [ir, ic])

    def normalize(self):
        ks_orbitals_normilize(self._data)

    def asarray(self):
        cdef cdouble[:, :, ::1] array = <cdouble[:self._data.ne, :self._data.grid.n[1],:self._data.grid.n[0]]>self._data.data
        return np.asarray(array)
