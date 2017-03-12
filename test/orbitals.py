import unittest

import tdse


class TestOrbitals(unittest.TestCase):
    def setUp(self):
        dr = 0.02
        r_max = 200
        Nr = int(r_max/dr)
        Nl = 60

        self.grid = tdse.grid.SGrid(Nr, Nl, r_max)
        self.atom = tdse.atom.Atom('Ar')
        self.orbs = self.atom.get_init_orbs(self.grid)
        self.field = tdse.field.SinField()

    def test_norm(self):
        self.orbs.norm()
