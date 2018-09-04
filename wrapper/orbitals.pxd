from types cimport cdouble, sh_f
from grid cimport cShGrid, cSpGrid, ShGrid
from atom cimport Atom, cAtom
from wavefunc cimport cShWavefunc
from sphere_harmonics cimport cYlmCache
from masks cimport cCoreMask

from libcpp.functional cimport function

from mpi4py.MPI cimport Comm
from mpi4py.libmpi cimport MPI_Comm


cdef extern from "orbitals.h":
    cdef cppclass cOrbitals "Orbitals":
        cAtom& atom
        cShGrid* grid
        cShWavefunc** wf
        cdouble* data
        MPI_Comm mpi_comm
        int mpi_rank
        cShWavefunc* mpi_wf

        cOrbitals(cAtom& atom, cShGrid* grid, MPI_Comm mpi_comm)
        void init()
        void setInitState(cdouble* data, int Nr, int Nl)

        double norm(sh_f mask)
        double norm()

        void norm_ne(double* n, sh_f mask)
        void norm_ne(double* n)

        void prod_ne(cOrbitals& orbs, cdouble* res)
        void normalize()

        double z()
        double z(sh_f mask)

        void z_ne(double* z, sh_f mask)
        void z_ne(double* z)

        double  n(cSpGrid* grid, int i[2], cYlmCache * ylm_cache)
        void n_sp(cSpGrid* grid, double* n, double* n_tmp, cYlmCache * ylm_cache)
        void n_l0(double* n, double* n_tmp)

        double cos(sh_f U)

        void ort()

cdef class Orbitals:
    cdef cOrbitals* cdata
    cdef Atom atom
    cdef Comm mpi_comm
    cdef ShGrid grid
