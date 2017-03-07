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

        self.grid = tdse.grid.SGrid(Nr, Nl, r_max)
        self.atom = tdse.atom.Atom('Ar')
        self.orbs = self.atom.get_init_orbs(self.grid)
        self.field = tdse.field.SinField()

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

        self.grid = tdse.grid.SGrid(Nr, Nl, r_max)
        self.atom = tdse.atom.Atom('Ar')
        self.orbs = self.atom.get_init_orbs(self.grid)
        self.uh = np.ndarray(Nr)
        self.ws = tdse.workspace.SOrbsWorkspace(grid=self.grid, atom=self.atom)
        self.field = tdse.field.SinField()
        self.sp_grid = tdse.grid.SpGrid(Nr, 32, 1, r_max)

    def time_hartree_potential_l0(self):
        tdse.hartree_potential.l0(self.orbs, self.uh)

    def time_lda(self):
        tdse.hartree_potential.lda(0, self.orbs, self.sp_grid, self.uh)

    def time_orbitals_propagate(self):
        self.ws.prop(self.orbs, self.field, 0.0, 0.1)


class WavefuncPropagate:
    timer = time.time

    def setup(self):
        dr = 0.002
        r_max = 200
        Nr = int(r_max/dr)
        Nl = 64

        self.grid = tdse.grid.SGrid(Nr, Nl, r_max)
        self.atom = tdse.atom.Atom('H')
        self.wf = tdse.atom.ground_state(self.grid)
        self.ws = tdse.workspace.SKnWorkspace(grid=self.grid, atom=self.atom)
        self.field = tdse.field.SinField()

    def time_propagate(self):
        self.ws.prop(self.wf, self.field, 0.0, 0.1)

    def time_z(self):
        self.wf.z()

    def time_norm(self):
        self.wf.norm()
