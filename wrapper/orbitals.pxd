from types cimport cdouble, sh_f
from grid cimport sh_grid_t, sp_grid_t
from wavefunc cimport sh_wavefunc_t
from sphere_harmonics cimport ylm_cache_t

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

    orbitals_t* orbials_new(int ne, sh_grid_t* grid, mpi.MPI_Comm mpi_comm)
    void orbitals_del(orbitals_t* orbs)
    double orbitals_norm(orbitals_t* orbs, sh_f mask)
    void orbitals_norm_ne(orbitals_t* orbs, double* n, sh_f mask)
    void orbitals_normalize(orbitals_t* orbs)
    double orbitals_n(orbitals_t* orbs, sp_grid_t* grid, int i[2], ylm_cache_t* ylm_cache)
    void orbitals_n_sp(orbitals_t* orbs, sp_grid_t* grid, double* n, double* n_tmp, ylm_cache_t* ylm_cache)
    void orbitals_set_init_state(orbitals_t* orbs, cdouble* data, int l_max)

cdef class SOrbitals:
    cdef orbitals_t* _data
