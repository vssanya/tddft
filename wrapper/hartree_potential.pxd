from wavefunc cimport sh_wavefunc_t
from orbitals cimport orbitals_t
from grid cimport sp_grid_t
from sphere_harmonics cimport ylm_cache_t

cdef extern from "hartree_potential.h":
    void hartree_potential(orbitals_t* orbs, int l, double* U, double* U_local, double* f, int order)
    void hartree_potential_wf_l0(sh_wavefunc_t* wf, double* U, double* f, int order)
    double mod_grad_n(sp_grid_t* grid, double* n, int ir, int ic)
    double ux_lda_func(double n)
    double uc_lda_func(double n)
    double uxc_lb_func(double n, double x)
    void uxc_lb(
        int l, orbitals_t* orbs,
        double* U,
        sp_grid_t* grid,
        double* n, # for calc using mpi
        double* n_local, # for calc using mpi
        ylm_cache_t* ylm_cache
    )
