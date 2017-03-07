import numpy as np
cimport numpy as np

from types cimport cdouble
from grid cimport SGrid, SpGrid
from wavefunc cimport sh_wavefunc_norm, swavefunc_from_point

cdef class SOrbitals:
    def __cinit__(self, int ne, SGrid grid):
        self._data = ks_orbials_new(ne, grid.data)

    def __dealloc__(self):
        orbitals_del(self._data)
    
    def n(self, SpGrid grid, int ir, int ic):
        return orbitals_n(self._data, grid.data, [ir, ic])

    def norm(self):
        return orbitals_norm(self._data)

    def normalize(self):
        orbitals_normalize(self._data)

    def get_wf(self, int ne):
        assert(ne < self._data.ne)
        return swavefunc_from_point(self._data.wf[ne])

    def asarray(self):
        cdef cdouble[:, :, ::1] array = <cdouble[:self._data.ne, :self._data.grid.n[1],:self._data.grid.n[0]]>self._data.data
        return np.asarray(array)

    def norm_ne(self):
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] res = np.ndarray(self._data.ne, dtype=np.double)
        for i in range(self._data.ne):
            res[i] = sh_wavefunc_norm(self._data.wf[i])

        return res
