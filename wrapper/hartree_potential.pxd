from orbitals cimport ks_orbitals_t
from grid cimport sp_grid_t

cdef extern from "hartree_potential.h":
    void hartree_potential_l0(ks_orbitals_t* orbs, double* U);
    void hartree_potential_l1(ks_orbitals_t* orbs, double* U);
    void hartree_potential_l2(ks_orbitals_t* orbs, double* U);
    void ux_lda(int l, ks_orbitals_t* orbs, double* U, sp_grid_t* sp_grid);
