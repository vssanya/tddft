from wavefunc cimport sh_wavefunc_t
from orbitals cimport orbitals_t
from grid cimport sh_grid_t
from types cimport sh_f

cdef extern from "atom.h":
    ctypedef void (*atom_init_f)(orbitals_t* orbs)
    ctypedef void (*atom_ort_f)(orbitals_t* orbs)
    ctypedef struct atom_t:
        int ne
        atom_init_f init
        atom_ort_f ort
        sh_f u
        sh_f dudz

    void atom_hydrogen_init(orbitals_t* orbs)
    double atom_hydrogen_sh_u(sh_grid_t* grid, int ir, int il, int m)
    double atom_hydrogen_sh_dudz(sh_grid_t* grid, int ir, int il, int m)
    void atom_hydrogen_ground(sh_wavefunc_t* wf)
    atom_t atom_hydrogen

    void atom_argon_init(orbitals_t* orbs)
    void atom_argon_ort(orbitals_t* orbs)
    double atom_argon_sh_u(sh_grid_t* grid, int ir, int il, int m)
    double atom_argon_sh_dudz(sh_grid_t* grid, int ir, int il, int m)
    atom_t atom_argon

    void atom_neon_init(orbitals_t* orbs)
    void atom_neon_ort(orbitals_t* orbs)
    double atom_neon_sh_u(sh_grid_t* grid, int ir, int il, int m)
    double atom_neon_sh_dudz(sh_grid_t* grid, int ir, int il, int m)
    atom_t atom_neon

cdef class Atom:
    cdef atom_t* _data
