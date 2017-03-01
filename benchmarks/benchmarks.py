import numpy as np
from tdse import grid, wavefunc, orbitals, field, atom, workspace, calc, utils, hartree_potential


class TimeSuite:
    def setup(self):
        dr = 0.02
        r_max = 200
        Nr = int(r_max/dr)
        Nl = 60

        self.grid = grid.SGrid(Nr, Nl, r_max)
        self.atom = atom.Atom('Ar')
        self.orbs = self.atom.get_init_orbs(self.grid)
        self.uh = np.ndarray(Nr)
        self.ws = workspace.SOrbsWorkspace(grid=self.grid, atom=self.atom)
        self.field = field.SinField()
        self.sp_grid = grid.SpGrid(Nr, 32, 1, r_max)

    def time_hartree_potential_l0(self):
        hartree_potential.l0(self.orbs, self.uh)

    def time_lda(self):
        hartree_potential.lda(0, self.orbs, self.sp_grid, self.uh)

    def time_orbitals_propagate(self):
        self.ws.prop(self.orbs, self.field, 0.0, 0.1)
