from wavefunc cimport sh_wavefunc_t
from orbitals cimport orbitals_t
from grid cimport sp_grid_t

cdef extern from "hartree_potential.h":
    void hartree_potential_l0(orbitals_t* orbs, double* U, double* f)
    void hartree_potential_wf_l0(sh_wavefunc_t* wf, double* U, double* f)
    void hartree_potential_l1(orbitals_t* orbs, double* U, double* f)
    void hartree_potential_l2(orbitals_t* orbs, double* U, double* f)
    void ux_lda(int l, orbitals_t* orbs, double* U, sp_grid_t* sp_grid, double* n)
    void ux_lda_n(int l, sp_grid_t* grid, double* n, double* U)
