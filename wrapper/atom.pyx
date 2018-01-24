import numpy as np
cimport numpy as np

from types cimport cdouble
from wavefunc cimport SWavefunc
from grid cimport ShGrid


cdef class Atom:
    @staticmethod
    cdef Atom from_c(const atom_t* atom):
        obj = <Atom>Atom.__new__(Atom)
        obj.cdata = atom[0]
        return obj

    @property
    def n_orbs(self):
        return self.cdata.n_orbs

    def get_l(self, int i):
        assert(i<self.cdata.n_orbs)
        return self.cdata.l[i]

    def get_m(self, int i):
        assert(i<self.cdata.n_orbs)
        return self.cdata.m[i]

H        = Atom.from_c(&atom_hydrogen)
H_smooth = Atom.from_c(&atom_hydrogen_smooth)
Ne       = Atom.from_c(&atom_neon)
Ar       = Atom.from_c(&atom_argon)
Ar_ion   = Atom.from_c(&atom_argon_ion)
Ar_sae   = Atom.from_c(&atom_argon_sae)
NONE = Atom.from_c(&atom_none)

cdef class HAtom:
    def __cinit__(self, int Z):
        self.cdata.Z = Z
        self.cdata.n_orbs = 1
        self.cdata.u = <pot_f>atom_u_coulomb
        self.cdata.dudz = <pot_f>atom_dudz_coulomb
        self.cdata.u_type = POTENTIAL_COULOMB

    def __init__(self, int Z):
        pass

def ground_state(ShGrid grid) -> SWavefunc:
    wf = SWavefunc(grid)
    atom_hydrogen_ground(wf.cdata)
    return wf
