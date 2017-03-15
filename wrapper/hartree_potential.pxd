from wavefunc cimport sh_wavefunc_t
from orbitals cimport orbitals_t
from grid cimport sp_grid_t
from sphere_harmonics cimport ylm_cache_t

cdef extern from "hartree_potential.h":
    void hartree_potential_l0(orbitals_t* orbs, double* U, double* U_local, double* f)
    void hartree_potential_wf_l0(sh_wavefunc_t* wf, double* U, double* f)
    void hartree_potential_l1(orbitals_t* orbs, double* U, double* f)
    void hartree_potential_l2(orbitals_t* orbs, double* U, double* f)
    void ux_lda(
        int l, orbitals_t* orbs,
        double* U,
        sp_grid_t* grid,
        double* n, # for calc using mpi
        double* n_local, # for calc using mpi
        ylm_cache_t* ylm_cache
    );
    void ux_lda_n(int l, sp_grid_t* grid, double* n, double* U, ylm_cache_t* ylm_cache)
