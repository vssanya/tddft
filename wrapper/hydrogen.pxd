from wavefunc cimport sphere_wavefunc_t
from orbitals cimport ks_orbitals_t
from grid cimport sh_grid_t

cdef extern from "hydrogen.h":
    double hydrogen_sh_u(sh_grid_t* grid, int ir, int il, int m)
    double hydrogen_sh_dudz(sh_grid_t* grid, int ir, int il, int m)
    void hydrogen_ground(sphere_wavefunc_t* wf)

cdef extern from "argon.h":
    void argon_init(ks_orbitals_t* orbs)
    void argon_ort(ks_orbitals_t* orbs)
