cdef extern from "mpi-compat.h":
    pass

from types cimport cdouble, sh_f
from grid cimport cShGrid, cShNeGrid, cSpGrid, ShGrid, ShNeGrid
from atom cimport Atom, cAtom
from wavefunc cimport Wavefunc
from sphere_harmonics cimport cYlmCache

from libcpp.functional cimport function

from mpi4py.MPI cimport Comm
from mpi4py.libmpi cimport MPI_Comm

cdef extern from "orbitals.h":
    cdef cppclass Orbitals[Grid]:
        cAtom& atom
        Grid& grid
        Wavefunc[Grid]** wf
        cdouble* data
        MPI_Comm mpi_comm
        int mpi_rank
        Wavefunc[Grid]* mpi_wf

        Orbitals(cAtom& atom, Grid& grid, MPI_Comm mpi_comm)
        void init()
        void init_shell(int shell)
        void setInitState(cdouble* data, int Nr, int Nl)

        double norm(sh_f mask)
        double norm()

        void norm_ne(double* n, sh_f mask)
        void norm_ne(double* n)

        void prod_ne(Orbitals& orbs, cdouble* res)
        void normalize()

        double z()
        double z(sh_f mask)

        void z_ne(double* z, sh_f mask)
        void z_ne(double* z)

        void collect(cdouble* dest, int Nl)

        double  n(cSpGrid* grid, int i[2], cYlmCache * ylm_cache)
        void n_sp(cSpGrid& grid, double* n, double* n_tmp, cYlmCache * ylm_cache)
        void n_l0(double* n, double* n_tmp)

        double cos(sh_f U)
        double cos(double* U)

        void ort()

    ctypedef Orbitals[cShGrid] cShOrbitals "ShOrbitals"
    ctypedef Orbitals[cShNeGrid] cShNeOrbitals "ShNeOrbitals"

cdef class ShOrbitals:
    cdef Orbitals[cShGrid]* cdata
    cdef Atom atom
    cdef Comm mpi_comm
    cdef ShGrid grid

cdef class ShNeOrbitals:
    cdef Orbitals[cShNeGrid]* cdata
    cdef Atom atom
    cdef Comm mpi_comm
    cdef ShNeGrid grid
