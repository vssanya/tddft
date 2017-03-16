import unittest

import tdse
import numpy as np


class TestOrbitals(unittest.TestCase):
    def setUp(self):
        dr = 0.02
        r_max = 200
        Nr = int(r_max/dr)
        Nl = 60

        self.grid = tdse.grid.ShGrid(Nr, Nl, r_max)
        self.sp_grid = tdse.grid.SpGrid(Nr, 32, 1, r_max)
        self.ylm_cache = tdse.sphere_harmonics.YlmCache(Nl, self.sp_grid)

        self.atom = tdse.atom.Atom('Ar')
        self.orbs = self.atom.get_init_orbs(self.grid)
        self.field = tdse.field.SinField()
        self.ws = tdse.workspace.SOrbsWorkspace(self.grid, self.sp_grid, atom=self.atom, ylm_cache=self.ylm_cache)

    # def test_norm(self):
        # self.orbs.norm()
        # print("Test Norm")

    def test_prop(self):
        self.ws.prop(self.orbs, self.field, 0.0, 0.1)

    def test_lda(self):
        n = self.orbs.n_sp(self.sp_grid, self.ylm_cache)
        ulda_1 = tdse.hartree_potential.lda(0, self.orbs, self.sp_grid, self.ylm_cache)
        ulda_2 = tdse.hartree_potential.lda_n(0, n, self.sp_grid, self.ylm_cache)
        self.assertAlmostEqual(np.sum(ulda_1 - ulda_2), 0.0)

    def test_az(self):
        tdse.calc.az(self.orbs, self.atom, self.field, 0.0)
