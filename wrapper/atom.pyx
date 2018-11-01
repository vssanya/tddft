import numpy as np
cimport numpy as np

from types cimport cdouble
from wavefunc cimport ShWavefunc
from grid cimport ShGrid

import tdse.utils
if tdse.utils.is_jupyter_notebook():
    import matplotlib.pyplot as plt
    from IPython.core.pylabtools import print_figure


cdef class AtomCache:
    def __cinit__(self, Atom atom, ShGrid grid, double[::1] u = None):
        self.atom = atom
        self.grid = grid
        if u is None:
            self.cdata = new cAtomCache(atom.cdata[0], grid.data)
        else:
            self.cdata = new cAtomCache(atom.cdata[0], grid.data, &u[0])

    def __init__(self, Atom atom, ShGrid grid, double[::1] u = None):
        pass

    def __dealloc__(self):
        del self.cdata

    @property
    def u(self):
        return np.asarray(<double[:self.grid.Nr]>(self.cdata.data_u))

    @property
    def dudz(self):
        return np.asarray(<double[:self.grid.Nr]>(self.cdata.data_dudz))

    def _figure_data(self, format):
        fig, ax = plt.subplots()
        fig.set_size_inches((6,3))

        ax.plot(self.grid.r, self.u)

        ax.set_xlabel('r, (a.u.)')
        ax.set_ylabel('U, (a.u.)')

        ax.set_yscale('log')

        data = print_figure(fig, format)
        plt.close(fig)
        return data

    def _repr_png_(self):
        return self._figure_data('png')

    def write_params(self, params_grp):
        subgrp = params_grp.create_group('atom')

        self.atom.write_params(subgrp)

        subgrp.create_dataset('u'   , (self.grid.Nr,), dtype="d")[:] = self.u
        subgrp.create_dataset('dudz', (self.grid.Nr,), dtype="d")[:] = self.dudz


cdef class State:
    @staticmethod
    cdef State from_c(cAtom.cState state):
        obj = <State>State.__new__(State)
        obj.cdata = state
        return obj

    @property
    def l(self):
        return self.cdata.l

    @property
    def m(self):
        return self.cdata.m

    @property
    def n(self):
        return self.cdata.n

    @property
    def countElectrons(self):
        return self.cdata.countElectrons


cdef class Atom:
    @staticmethod
    cdef Atom from_c(cAtom* atom, str name):
        obj = <Atom>Atom.__new__(Atom)
        obj.cdata = atom
        obj.name = name
        obj.ground_state = State.from_c(atom.groundState)
        return obj

    def __dealloc__(self):
        print("Free atom")
        del self.cdata

    @property
    def countOrbs(self):
        return self.cdata.countOrbs

    @property
    def countElectrons(self):
        return self.cdata.countElectrons

    @property
    def l_max(self):
        cdef int lmax = 0
        cdef int i = 0

        for i in range(self.cdata.countOrbs):
            if lmax < self.cdata.orbs[i].l:
                lmax = self.cdata.orbs[i].l

        return lmax

    def get_l(self, int i):
        assert(i<self.cdata.countOrbs)
        return self.cdata.orbs[i].l

    def get_m(self, int i):
        assert(i<self.cdata.countOrbs)
        return self.cdata.orbs[i].m

    def getCountElectrons(self, int i):
        return self.cdata.orbs[i].countElectrons

    @property
    def Z(self):
        return self.cdata.Z

    def u(self, double r):
        return self.cdata.u(r)

    def dudz(self, double r):
        return self.cdata.dudz(r)

    def write_params(self, params_grp):
        params_grp.attrs['type'] = self.name


H        = Atom.from_c(<cAtom*> new HAtom(), "H")
H_smooth = Atom.from_c(<cAtom*> new HSmothAtom(), "H_smooth")

Ne       = Atom.from_c(<cAtom*> new NeAtom(), "Ne")
Ar       = Atom.from_c(<cAtom*> new ArAtom(), "Ar")
Kr       = Atom.from_c(<cAtom*> new KrAtom(), "Kr")
Ar_sae   = Atom.from_c(<cAtom*> new ArSaeAtom(), "Ar_sae")
Ar_sae_smooth = Atom.from_c(<cAtom*> new ArSaeSmoothAtom(), "Ar_sae_smooth")

Fm = Atom.from_c(<cAtom*> new FNegativeIon(), "F-")

Na = Atom.from_c(<cAtom*> new NaAtom(), "Na")
Na_sae   = Atom.from_c(<cAtom*> new NaAtomSGB(), "Na_sae")

Mg = Atom.from_c(<cAtom*> new MgAtom(), "Mg")

NONE = Atom.from_c(<cAtom*> new NoneAtom(), "None")
