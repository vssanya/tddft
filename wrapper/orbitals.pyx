import numpy as np
cimport numpy as np

from types cimport cdouble
from abs_pot cimport mask_core
from grid cimport ShGrid, SpGrid
from atom cimport Atom
from wavefunc cimport sh_wavefunc_norm, swavefunc_from_point, sh_wavefunc_cos_r
from sphere_harmonics cimport YlmCache


from mpi4py.libmpi cimport MPI_COMM_NULL, MPI_Gather, MPI_C_DOUBLE_COMPLEX
from mpi4py.MPI import COMM_NULL


cdef class SOrbitals:
    def __cinit__(self, Atom atom, ShGrid grid, Comm comm = None):
        if comm is None:
            comm = COMM_NULL

        self.cdata = orbials_new(atom.cdata, grid.data, comm.ob_mpi)

        self.mpi_comm = comm
        self.grid = grid
        self.atom = atom

    def __dealloc__(self):
        if self.cdata != NULL:
            orbitals_del(self.cdata)

    def init(self):
        orbitals_init(self.cdata)

    def ort(self):
        orbitals_ort(self.cdata)

    def load(self, filename):
        arr = self.asarray()
        arr[:] = 0.0

        cdef np.ndarray[np.complex_t, ndim=3] data
        cdef cdouble* data_ptr = NULL

        if not self.is_mpi() or self.cdata.mpi_rank == 0:
            data = np.load(filename)
            shape = [data.shape[1], data.shape[2]]
            data_ptr = <cdouble*>data.data
        else:
            shape = None

        if self.is_mpi():
            shape = self.mpi_comm.bcast(shape)

        orbitals_set_init_state(self.cdata, data_ptr, shape[1], shape[0])

    def n(self, SpGrid grid, YlmCache ylm_cache, int ir, int ic):
        return orbitals_n(self.cdata, grid.data, [ir, ic], ylm_cache.cdata)

    def z(self):
        return orbitals_z(self.cdata)

    def n_sp(self, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=2] n = None):
        cdef np.ndarray[np.double_t, ndim=2, mode='c'] n_local = np.ndarray(grid.shape, dtype=np.double)
        cdef double* n_ptr = NULL

        if self.is_root():
            n = np.ndarray(grid.shape, dtype=np.double)
            n_ptr = <double*>n.data

        orbitals_n_sp(self.cdata, grid.data, n_ptr, <double*>n_local.data, ylm_cache.cdata) 

        return n

    def norm(self, masked=False):
        if masked:
            return orbitals_norm(self.cdata, mask_core)
        else:
            return orbitals_norm(self.cdata, NULL)

    def norm_ne(self, np.ndarray[double, ndim=1, mode='c'] norm = None, masked=False):
        cdef double* res_ptr = NULL
        if self.is_root():
            if norm is None:
                norm = np.ndarray(self.cdata.atom.n_orbs, dtype=np.double)
            res_ptr = <double*>norm.data

        if masked:
            orbitals_norm_ne(self.cdata, res_ptr, mask_core)
        else:
            orbitals_norm_ne(self.cdata, res_ptr, NULL)

        return norm


    def normalize(self):
        orbitals_normalize(self.cdata)

    def get_wf(self, int ie):
        assert(ie < self.cdata.atom.n_orbs)
        return swavefunc_from_point(self.cdata.wf[ie])

    def asarray(self):
        cdef cdouble[:, :, ::1] res
        if self.is_mpi():
            array = <cdouble[:1, :self.cdata.grid.n[1],:self.cdata.grid.n[0]]>self.cdata.data
        else:
            array = <cdouble[:self.cdata.atom.n_orbs, :self.cdata.grid.n[1],:self.cdata.grid.n[0]]>self.cdata.data

        return np.asarray(array)

    def is_mpi(self) -> bool:
        return self.cdata.mpi_comm != MPI_COMM_NULL

    def is_root(self) -> bool:
        return not self.is_mpi() or self.cdata.mpi_rank == 0

    def load(self, file):
        pass

    def save(self, file):
        cdef np.ndarray[np.complex_t, ndim=3, mode='c'] arr = None
        cdef cdouble* arr_ptr = NULL
        cdef int size = self.cdata.grid.n[0]*self.cdata.grid.n[1]

        if self.is_mpi():
            if self.cdata.mpi_rank == 0:
                arr = np.ndarray((self.cdata.atom.n_orbs, self.cdata.grid.n[1], self.cdata.grid.n[0]), dtype=np.complex)
                arr_ptr = <cdouble*>arr.data
            MPI_Gather(self.cdata.data, size, MPI_C_DOUBLE_COMPLEX, arr_ptr, size, MPI_C_DOUBLE_COMPLEX, 0, self.cdata.mpi_comm)
            if self.cdata.mpi_rank == 0:
                np.save(file, self.asarray())
        else:
            np.save(file, self.asarray())

    def __str__(self):
        return "Orbitals MPI, wf.m = {}".format(self.cdata.mpi_wf.m)

    def grad_u(self, ie=0):
        cdef np.ndarray[np.double_t, ndim=1] res = np.ndarray(self.cdata.grid.n[0], np.double)

        sh_wavefunc_cos_r(self.cdata.wf[ie], self.atom.cdata.dudz, <double*>res.data)
        return res

