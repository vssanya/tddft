import time
import numpy as np

import tdse


class OrbitalsFunc:
    timer = time.time

    def setup(self):
        dr = 0.02
        r_max = 200
        Nr = int(r_max/dr)
        Nl = 60

        self.grid = tdse.grid.ShGrid(Nr, Nl, r_max)
        self.atom = tdse.atom.Atom('Ar')
        self.orbs = tdse.orbitals.SOrbitals(self.atom, self.grid)
        self.orbs.init()
        self.field = tdse.field.TwoColorSinField()

    def time_norm(self):
        self.orbs.norm()

    def time_orbs_az(self):
        tdse.calc.az(self.orbs, self.atom, self.field, 0.0)

    def time_ionization_prob(self):
        tdse.calc.ionization_prob(self.orbs)


class OrbitalsPropagate:
    timer = time.time

    def setup(self):
        dr = 0.02
        r_max = 200
        Nr = int(r_max/dr)
        Nl = 60

        self.grid = tdse.grid.ShGrid(Nr, Nl, r_max)
        self.sp_grid = tdse.grid.SpGrid(Nr, 32, 1, r_max)
        self.n = np.ndarray((32, Nr))

        self.ylm_cache = tdse.sphere_harmonics.YlmCache(Nl, self.sp_grid)

        self.atom = tdse.atom.Atom('Ar')
        self.orbs = tdse.orbitals.SOrbitals(self.atom, self.grid)
        self.orbs.init()
        self.uh = np.ndarray(Nr)
        self.uabs = tdse.abs_pot.UabsMultiHump(0.1, 10)
        self.ws = tdse.workspace.SOrbsWorkspace(self.grid, self.sp_grid, self.uabs, ylm_cache=self.ylm_cache)
        self.field = tdse.field.TwoColorSinField()

    def time_hartree_potential_l0(self):
        tdse.hartree_potential.l0(self.orbs, self.uh)

    def time_lda(self):
        tdse.hartree_potential.lda(0, self.orbs, self.sp_grid, self.ylm_cache, self.uh)

    def time_orbitals_propagate(self):
        self.ws.prop(self.orbs, self.atom, self.field, 0.0, 0.1)


class WavefuncPropagate:
    timer = time.time

    def setup(self):
        dr = 0.02
        r_max = 200
        Nr = int(r_max/dr)
        Nl = 64

        self.grid = tdse.grid.ShGrid(Nr, Nl, r_max)
        self.sp_grid = tdse.grid.SpGrid(Nr, 32, 1, r_max)
        self.ylm_cache = tdse.sphere_harmonics.YlmCache(Nl, self.sp_grid)
        self.atom = tdse.atom.Atom('H')
        self.n = np.ndarray((Nr, 32))
        self.wf = tdse.atom.ground_state(self.grid)
        self.uabs = tdse.abs_pot.UabsMultiHump(0.1, 10)
        self.ws = tdse.workspace.SKnWorkspace(grid=self.grid, uabs=self.uabs)
        self.field = tdse.field.TwoColorSinField()

    def time_propagate(self):
        self.ws.prop(self.wf, self.atom, self.field, 0.0, 0.1)

    def time_z(self):
        self.wf.z()

    def time_az(self):
        tdse.calc.az(self.wf, self.atom, self.field, 0.0)

    def time_n_sp(self):
        self.wf.n_sp(self.sp_grid, self.ylm_cache, self.n)

    def time_norm(self):
        self.wf.norm()
