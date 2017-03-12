from types cimport cdouble
from grid cimport sh_grid_t, sp_grid_t
from wavefunc cimport sh_wavefunc_t

cimport mpi4py.libmpi as mpi

cdef extern from "orbitals.h":
    ctypedef struct orbitals_t:
        int ne
        sh_grid_t* grid
        sh_wavefunc_t** wf
        cdouble* data
        mpi.MPI_Comm mpi_comm
        int mpi_rank
        sh_wavefunc_t* mpi_wf

    orbitals_t* ks_orbials_new(int ne, sh_grid_t* grid, mpi.MPI_Comm mpi_comm)
    void orbitals_del(orbitals_t* orbs)
    double orbitals_norm(orbitals_t* orbs)
    void orbitals_normalize(orbitals_t* orbs)
    double orbitals_n(orbitals_t* orbs, sp_grid_t* grid, int i[2])
    void orbitals_n_sp(orbitals_t* orbs, sp_grid_t* grid, double* n)

cdef class SOrbitals:
    cdef orbitals_t* _data
