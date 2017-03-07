from orbitals cimport orbitals_t
from grid cimport sp_grid_t

cdef extern from "hartree_potential.h":
    void hartree_potential_l0(orbitals_t* orbs, double* U, double* f);
    void hartree_potential_l1(orbitals_t* orbs, double* U, double* f);
    void hartree_potential_l2(orbitals_t* orbs, double* U, double* f);
    void ux_lda(int l, orbitals_t* orbs, double* U, sp_grid_t* sp_grid);
