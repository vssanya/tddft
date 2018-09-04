import numpy as np
cimport numpy as np

from libcpp.functional cimport function

from mpi4py.libmpi cimport MPI_COMM_NULL, MPI_Gather, MPI_C_DOUBLE_COMPLEX
from mpi4py.MPI import COMM_NULL

from types cimport cdouble, complex_t, sh_f
from abs_pot cimport mask_core
from grid cimport ShGrid, SpGrid
from atom cimport Atom
from wavefunc cimport swavefunc_from_point
from sphere_harmonics cimport YlmCache
from masks cimport CoreMask


cdef class Orbitals:
    def __cinit__(self, Atom atom, ShGrid grid, Comm comm = None):
        if comm is None:
            comm = COMM_NULL

        self.cdata = new cOrbitals(atom.cdata[0], grid.data, comm.ob_mpi)

        self.mpi_comm = comm
        self.grid = grid
        self.atom = atom

    def __dealloc__(self):
        if self.cdata != NULL:
            del self.cdata

    def init(self):
        self.cdata.init()

    def ort(self):
        self.cdata.ort()

    def is_root_or_not_mpi(self):
        return not self.is_mpi() or self.cdata.mpi_rank == 0

    def load(self, filename):
        arr = self.asarray()
        arr[:] = 0.0

        cdef np.ndarray[np.complex_t, ndim=3] data
        cdef cdouble* data_ptr = NULL

        if self.is_root_or_not_mpi():
            data = np.load(filename)
            shape = [data.shape[1], data.shape[2]]
            data_ptr = <cdouble*>data.data
        else:
            shape = None

        if self.is_mpi():
            shape = self.mpi_comm.bcast(shape)

        self.cdata.setInitState(<cdouble*>data_ptr, shape[1], shape[0])

    @property
    def shape(self):
        return (self.cdata.atom.countOrbs, self.cdata.grid.n[1], self.cdata.grid.n[0])

    def save(self, filename):
        data = self.asarray()

        if self.is_mpi():
            if self.mpi_rank == 0:
                data_save = np.zeros(self.shape, np.complex)
            else:
                data_save = None

            self.mpi_comm.Gather(data, data_save, root=0)
        else:
            data_save = data

        if self.is_root_or_not_mpi():
            np.save(filename, data_save)

    def n(self, SpGrid grid, YlmCache ylm_cache, int ir, int ic):
        return self.cdata.n(grid.data, [ir, ic], ylm_cache.cdata)

    def z(self, CoreMask mask = None):
        if mask is None:
            return self.cdata.z()
        else:
            return self.cdata.z(<sh_f> mask.cdata[0])

    def z_ne(self, CoreMask mask = None, np.ndarray[double, ndim=1] z = None):
        cdef double* res_ptr = NULL
        if self.is_root():
            if z is None:
                z = np.ndarray(self.atom.countOrbs, dtype=np.double)
            res_ptr = <double*>z.data

        if mask is None:
            self.cdata.z_ne(res_ptr)
        else:
            self.cdata.z_ne(res_ptr, <sh_f> mask.cdata[0])

        return z

    def n_sp(self, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=2] n = None):
        cdef np.ndarray[np.double_t, ndim=2, mode='c'] n_local = np.ndarray(grid.shape, dtype=np.double)
        cdef double* n_ptr = NULL

        if self.is_root():
            n = np.ndarray(grid.shape, dtype=np.double)
            n_ptr = <double*>n.data

        self.cdata.n_sp(grid.data, n_ptr, <double*>n_local.data, ylm_cache.cdata)

        return n

    def n_l0(self, np.ndarray[np.double_t, ndim=1] n = None):
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] n_local = np.ndarray(self.cdata.grid.n[0], dtype=np.double)
        cdef double* n_ptr = NULL

        if self.is_root():
            n = np.ndarray(self.cdata.grid.n[0], dtype=np.double)
            n_ptr = <double*>n.data

        self.cdata.n_l0(n_ptr, <double*>n_local.data)

        return n

    def norm(self, CoreMask mask = None):
        if mask is not None:
            return self.cdata.norm(<sh_f> mask.cdata[0])
        else:
            return self.cdata.norm()

    def norm_ne(self, np.ndarray[double, ndim=1, mode='c'] norm = None, CoreMask mask = None):
        cdef double* res_ptr = NULL
        if self.is_root():
            if norm is None:
                norm = np.ndarray(self.cdata.atom.countOrbs, dtype=np.double)
            res_ptr = <double*>norm.data

        if mask is not None:
            self.cdata.norm_ne(res_ptr, <sh_f> mask.cdata[0])
        else:
            self.cdata.norm_ne(res_ptr)

        return norm

    def prod_ne(self, Orbitals orbs, np.ndarray[complex_t, ndim=1, mode='c'] res = None):
        cdef cdouble* res_ptr = NULL
        if self.is_root():
            if res is None:
                res = np.ndarray(self.cdata.atom.countOrbs, dtype=np.double)
            res_ptr = <cdouble*>res.data

        self.cdata.prod_ne(orbs.cdata[0], res_ptr)

        return res

    def normalize(self):
        self.cdata.normalize()

    def get_wf(self, int ie):
        assert(ie < self.cdata.atom.countOrbs)
        return swavefunc_from_point(self.cdata.wf[ie], self.grid)

    def asarray(self):
        cdef complex_t[:, :, ::1] res
        if self.is_mpi():
            array = <complex_t[:1, :self.cdata.grid.n[1],:self.cdata.grid.n[0]]>(<complex_t*>self.cdata.data)
        else:
            array = <complex_t[:self.cdata.atom.countOrbs, :self.cdata.grid.n[1],:self.cdata.grid.n[0]]>(<complex_t*>self.cdata.data)

        return np.asarray(array)

    def is_mpi(self) -> bool:
        return self.cdata.mpi_comm != MPI_COMM_NULL

    def is_root(self) -> bool:
        return not self.is_mpi() or self.cdata.mpi_rank == 0

    def save(self, filename):
        cdef np.ndarray[np.complex_t, ndim=3, mode='c'] arr = None
        cdef cdouble* arr_ptr = NULL
        cdef int size = self.cdata.grid.n[0]*self.cdata.grid.n[1]

        if self.is_mpi():
            if self.cdata.mpi_rank == 0:
                arr = np.ndarray((self.cdata.atom.countOrbs, self.cdata.grid.n[1], self.cdata.grid.n[0]), dtype=np.complex)
                arr_ptr = <cdouble*>arr.data
            MPI_Gather(self.cdata.data, size, MPI_C_DOUBLE_COMPLEX, arr_ptr, size, MPI_C_DOUBLE_COMPLEX, 0, self.cdata.mpi_comm)
            if self.cdata.mpi_rank == 0:
                np.save(filename, self.asarray())
        else:
            np.save(filename, self.asarray())

    def __str__(self):
        return "Orbitals MPI, wf.m = {}".format(self.cdata.mpi_wf.m)

    # def grad_u(self, ie=0):
        # cdef np.ndarray[np.double_t, ndim=1] res = np.ndarray(self.cdata.grid.n[0], np.double)

        # sh_wavefunc_cos_r(self.cdata.wf[ie], <sh_f>self.atom.cdata.dudz, <double*>res.data)
        # return res

