from types cimport cdouble
from grid cimport sh_grid_t, sp_grid_t
from wavefunc cimport sh_wavefunc_t

cdef extern from "orbitals.h":
    ctypedef struct orbitals_t:
        int ne
        sh_grid_t* grid
        sh_wavefunc_t** wf
        cdouble* data

    orbitals_t* ks_orbials_new(int ne, sh_grid_t* grid)
    void orbitals_del(orbitals_t* orbs)
    double orbitals_norm(orbitals_t* orbs)
    void orbitals_normalize(orbitals_t* orbs)
    double orbitals_n(orbitals_t* orbs, sp_grid_t* grid, int i[2])

cdef class SOrbitals:
    cdef orbitals_t* _data
