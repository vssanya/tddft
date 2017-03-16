import numpy as np
cimport numpy as np

import mpi4py.MPI
cimport mpi4py.MPI
cimport mpi4py.libmpi

from types cimport cdouble
from grid cimport ShGrid, SpGrid
from wavefunc cimport sh_wavefunc_norm, swavefunc_from_point
from sphere_harmonics cimport YlmCache


cdef class SOrbitals:
    def __cinit__(self, int ne, ShGrid grid, mpi4py.MPI.Comm comm = None):
        if comm is None:
            comm = mpi4py.MPI.COMM_NULL

        self._data = orbials_new(ne, grid.data, comm.ob_mpi)

    def __dealloc__(self):
        if self._data != NULL:
            orbitals_del(self._data)
    
    def n(self, SpGrid grid, YlmCache ylm_cache, int ir, int ic):
        return orbitals_n(self._data, grid.data, [ir, ic], ylm_cache._data)

    def n_sp(self, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=2] n = None):
        cdef np.ndarray[np.double_t, ndim=2, mode='c'] n_local = np.ndarray(grid.shape, dtype=np.double)
        cdef double* n_ptr = NULL

        if not self.is_mpi() or self._data.mpi_rank == 0:
            n = np.ndarray(grid.shape, dtype=np.double)
            n_ptr = <double*>n.data

        orbitals_n_sp(self._data, grid.data, n_ptr, <double*>n_local.data, ylm_cache._data) 

        return n

    def norm(self):
        return orbitals_norm(self._data)

    def normalize(self):
        orbitals_normalize(self._data)

    def get_wf(self, int ne):
        assert(ne < self._data.ne)
        return swavefunc_from_point(self._data.wf[ne])

    def asarray(self):
        cdef cdouble[:, :, ::1] res
        if self.is_mpi():
            array = <cdouble[:1, :self._data.grid.n[1],:self._data.grid.n[0]]>self._data.data
        else:
            array = <cdouble[:self._data.ne, :self._data.grid.n[1],:self._data.grid.n[0]]>self._data.data

        return np.asarray(array)

    def is_mpi(self) -> bool:
        return self._data.mpi_comm != mpi4py.libmpi.MPI_COMM_NULL

    def norm_ne(self):
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] res = np.ndarray(self._data.ne, dtype=np.double)
        for i in range(self._data.ne):
            res[i] = sh_wavefunc_norm(self._data.wf[i])

        return res

    def load(self, file):
        pass

    def save(self, file):
        cdef np.ndarray[np.complex_t, ndim=3, mode='c'] arr = None
        cdef cdouble* arr_ptr = NULL
        cdef int size = self._data.grid.n[0]*self._data.grid.n[1]

        if self.is_mpi():
            if self._data.mpi_rank == 0:
                arr = np.ndarray((self._data.ne, self._data.grid.n[1], self._data.grid.n[0]), dtype=np.complex)
                arr_ptr = <cdouble*>arr.data
            mpi4py.libmpi.MPI_Gather(self._data.data, size, mpi4py.libmpi.MPI_C_DOUBLE_COMPLEX, arr_ptr, size, mpi4py.libmpi.MPI_C_DOUBLE_COMPLEX, 0, self._data.mpi_comm)
            if self._data.mpi_rank == 0:
                np.save(file, self.asarray())
        else:
            np.save(file, self.asarray())

    def __str__(self):
        return "Orbitals MPI, wf.m = {}".format(self._data.mpi_wf.m)
