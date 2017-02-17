from orbitals cimport ks_orbitals_t

cdef extern from "hartree_potential.h":
    void hartree_potential_l0(ks_orbitals_t* orbs, double* U);
    void hartree_potential_l1(ks_orbitals_t* orbs, double* U);
    void hartree_potential_l2(ks_orbitals_t* orbs, double* U);
