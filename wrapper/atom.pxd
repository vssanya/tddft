from wavefunc cimport sh_wavefunc_t
from grid cimport sh_grid_t
from types cimport sh_f

cdef extern from "atom.h":
    ctypedef enum potential_type_e:
        POTENTIAL_SMOOTH, POTENTIAL_COULOMB

    ctypedef struct atom_t:
        int Z
        int n_orbs
        int* m
        int* l
        int* e_n
        sh_f u
        sh_f dudz
        potential_type_e u_type

    double atom_u_coulomb(atom_t* atom, sh_grid_t* grid, int ir)
    double atom_dudz_coulomb(atom_t* atom, sh_grid_t* grid, int ir)
    double atom_u_smooth(atom_t* atom, sh_grid_t* grid, int ir)
    double atom_dudz_smooth(atom_t* atom, sh_grid_t* grid, int ir)

    void atom_hydrogen_ground(sh_wavefunc_t* wf)

    atom_t atom_hydrogen
    atom_t atom_hydrogen_smooth

    atom_t atom_argon
    atom_t atom_argon_ion

    atom_t atom_neon

cdef class Atom:
    cdef atom_t* cdata
    @staticmethod
    cdef Atom from_c(atom_t* atom)
