import numpy as np
cimport numpy as np

from types cimport cdouble
from grid cimport SGrid, SpGrid
from wavefunc cimport sphere_wavefunc_norm

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

    def norm_ne(self):
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] res = np.ndarray(self._data.ne, dtype=np.double)
        for i in range(self._data.ne):
            res[i] = sphere_wavefunc_norm(self._data.wf[i])

        return res
