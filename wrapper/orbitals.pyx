import numpy as np
cimport numpy as np

import mpi4py.MPI
cimport mpi4py.MPI
cimport mpi4py.libmpi

from types cimport cdouble
from abs_pot cimport mask_core
from grid cimport ShGrid, SpGrid
from atom cimport Atom
from wavefunc cimport sh_wavefunc_norm, swavefunc_from_point
from sphere_harmonics cimport YlmCache


cdef class SOrbitals:
    def __cinit__(self, Atom atom, ShGrid grid, mpi4py.MPI.Comm comm = None):
        if comm is None:
            comm = mpi4py.MPI.COMM_NULL

        self._data = orbials_new(atom._data, grid.data, comm.ob_mpi)

        self.mpi_comm = comm
        self.grid = grid
        self.atom = atom

    def __dealloc__(self):
        if self._data != NULL:
            orbitals_del(self._data)

    def init(self):
        orbitals_init(self._data)

    def ort(self):
        self._data.atom.ort(self._data)

    def load(self, filename):
        arr = self.asarray()
        arr[:] = 0.0

        cdef np.ndarray[np.complex_t, ndim=3] data
        cdef cdouble* data_ptr = NULL

        if not self.is_mpi() or self._data.mpi_rank == 0:
            data = np.load(filename)
            shape = [data.shape[1], data.shape[2]]
            data_ptr = <cdouble*>data.data
        else:
            shape = None

        if self.is_mpi():
            shape = self.mpi_comm.bcast(shape)

        orbitals_set_init_state(self._data, data_ptr, shape[1], shape[0])
    
    def n(self, SpGrid grid, YlmCache ylm_cache, int ir, int ic):
        return orbitals_n(self._data, grid.data, [ir, ic], ylm_cache._data)

    def z(self):
        return orbitals_z(self._data)

    def n_sp(self, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=2] n = None):
        cdef np.ndarray[np.double_t, ndim=2, mode='c'] n_local = np.ndarray(grid.shape, dtype=np.double)
        cdef double* n_ptr = NULL

        if self.is_root():
            n = np.ndarray(grid.shape, dtype=np.double)
            n_ptr = <double*>n.data

        orbitals_n_sp(self._data, grid.data, n_ptr, <double*>n_local.data, ylm_cache._data) 

        return n

    def norm(self, masked=False):
        if masked:
            return orbitals_norm(self._data, mask_core)
        else:
            return orbitals_norm(self._data, NULL)

    def norm_ne(self, np.ndarray[double, ndim=1, mode='c'] norm = None, masked=False):
        cdef double* res_ptr = NULL
        if self.is_root():
            if norm is None:
                norm = np.ndarray(self._data.atom.n_orbs, dtype=np.double)
            res_ptr = <double*>norm.data

        if masked:
            orbitals_norm_ne(self._data, res_ptr, mask_core)
        else:
            orbitals_norm_ne(self._data, res_ptr, NULL)

        return norm


    def normalize(self):
        orbitals_normalize(self._data)

    def get_wf(self, int ie):
        assert(ie < self._data.atom.n_orbs)
        return swavefunc_from_point(self._data.wf[ie])

    def asarray(self):
        cdef cdouble[:, :, ::1] res
        if self.is_mpi():
            array = <cdouble[:1, :self._data.grid.n[1],:self._data.grid.n[0]]>self._data.data
        else:
            array = <cdouble[:self._data.atom.n_orbs, :self._data.grid.n[1],:self._data.grid.n[0]]>self._data.data

        return np.asarray(array)

    def is_mpi(self) -> bool:
        return self._data.mpi_comm != mpi4py.libmpi.MPI_COMM_NULL

    def is_root(self) -> bool:
        return not self.is_mpi() or self._data.mpi_rank == 0

    def load(self, file):
        pass

    def save(self, file):
        cdef np.ndarray[np.complex_t, ndim=3, mode='c'] arr = None
        cdef cdouble* arr_ptr = NULL
        cdef int size = self._data.grid.n[0]*self._data.grid.n[1]

        if self.is_mpi():
            if self._data.mpi_rank == 0:
                arr = np.ndarray((self._data.atom.n_orbs, self._data.grid.n[1], self._data.grid.n[0]), dtype=np.complex)
                arr_ptr = <cdouble*>arr.data
            mpi4py.libmpi.MPI_Gather(self._data.data, size, mpi4py.libmpi.MPI_C_DOUBLE_COMPLEX, arr_ptr, size, mpi4py.libmpi.MPI_C_DOUBLE_COMPLEX, 0, self._data.mpi_comm)
            if self._data.mpi_rank == 0:
                np.save(file, self.asarray())
        else:
            np.save(file, self.asarray())

    def __str__(self):
        return "Orbitals MPI, wf.m = {}".format(self._data.mpi_wf.m)
