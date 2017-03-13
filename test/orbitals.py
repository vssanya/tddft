import unittest

import tdse


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
