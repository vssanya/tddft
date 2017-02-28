from types cimport cdouble
from grid cimport sh_grid_t, sp_grid_t
from wavefunc cimport sphere_wavefunc_t

cdef extern from "ks_orbitals.h":
    ctypedef struct ks_orbitals_t:
        int ne
        sh_grid_t* grid
        sphere_wavefunc_t** wf
        cdouble* data

    ks_orbitals_t* ks_orbials_new(int ne, sh_grid_t* grid)
    void ks_orbitals_del(ks_orbitals_t* orbs)
    double ks_orbitals_norm(ks_orbitals_t* orbs)
    void ks_orbitals_normilize(ks_orbitals_t* orbs)
    double ks_orbitals_n(ks_orbitals_t* orbs, sp_grid_t* grid, int i[2])

cdef class SOrbitals:
    cdef ks_orbitals_t* _data
