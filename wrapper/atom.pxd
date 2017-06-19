from wavefunc cimport sh_wavefunc_t
from grid cimport sh_grid_t
from types cimport sh_f

cdef extern from "atom.h":
    ctypedef void (*atom_ort_f)(void* orbs)

    ctypedef enum potential_type_e:
        POTENTIAL_SMOOTH, POTENTIAL_COULOMB

    ctypedef struct atom_t:
        int Z
        int n_orbs
        int* m
        int* l
        int* e_n
        atom_ort_f ort
        sh_f u
        sh_f dudz
        potential_type_e u_type

    double atom_hydrogen_sh_u(sh_grid_t* grid, int ir, int il, int m)
    double atom_hydrogen_sh_dudz(sh_grid_t* grid, int ir, int il, int m)
    double atom_hydrogen_sh_u_smooth(sh_grid_t* grid, int ir, int il, int m)
    double atom_hydrogen_sh_dudz_smooth(sh_grid_t* grid, int ir, int il, int m)
    void atom_hydrogen_ground(sh_wavefunc_t* wf)
    atom_t atom_hydrogen
    atom_t atom_hydrogen_smooth

    double atom_argon_sh_u(sh_grid_t* grid, int ir, int il, int m)
    double atom_argon_sh_dudz(sh_grid_t* grid, int ir, int il, int m)
    atom_t atom_argon
    atom_t atom_argon_gs

    double atom_neon_sh_u(sh_grid_t* grid, int ir, int il, int m)
    double atom_neon_sh_dudz(sh_grid_t* grid, int ir, int il, int m)
    atom_t atom_neon

cdef class Atom:
    cdef atom_t* cdata
