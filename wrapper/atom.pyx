import numpy as np
cimport numpy as np

from types cimport cdouble
from wavefunc cimport ShWavefunc
from grid cimport ShGrid


cdef class AtomCache:
    def __cinit__(self, Atom atom, ShGrid grid):
        self.atom = atom
        self.grid = grid
        self.cdata = new cAtomCache(atom.cdata[0], grid.data)

    def __dealloc__(self):
        del self.cdata

    def get_u(self):
        cdef double[::1] array = <double[:self.cdata.grid.n[0]]>(self.cdata.data_u)
        return np.asarray(array)

    def get_dudz(self):
        cdef double[::1] array = <double[:self.cdata.grid.n[0]]>(self.cdata.data_dudz)
        return np.asarray(array)


cdef class Atom:
    @staticmethod
    cdef Atom from_c(cAtom* atom):
        obj = <Atom>Atom.__new__(Atom)
        obj.cdata = atom
        return obj

    def __dealloc__(self):
        del self.cdata

    @property
    def countOrbs(self):
        return self.cdata.countOrbs

    @property
    def l_max(self):
        cdef int lmax = 0
        cdef int i = 0

        for i in range(self.cdata.countOrbs):
            if lmax < self.cdata.orbs[i].l:
                l = self.cdata.orbs[i].l

        return l

    def get_l(self, int i):
        assert(i<self.cdata.countOrbs)
        return self.cdata.orbs[i].l

    def get_m(self, int i):
        assert(i<self.cdata.countOrbs)
        return self.cdata.orbs[i].m

    def u(self, double r):
        return self.cdata.u(r)

    def dudz(self, double r):
        return self.cdata.dudz(r)

H        = Atom.from_c(<cAtom*> new HAtom())
H_smooth = Atom.from_c(<cAtom*> new HSmothAtom())

Ar       = Atom.from_c(<cAtom*> new ArAtom())
Ar_sae   = Atom.from_c(<cAtom*> new ArSaeAtom())
Ar_sae_smooth = Atom.from_c(<cAtom*> new ArSaeSmoothAtom())

Na = Atom.from_c(<cAtom*> new NaAtom())
Na_sae   = Atom.from_c(<cAtom*> new NaAtomSGB())

NONE = Atom.from_c(<cAtom*> new NoneAtom())
