import unittest

import tdse
import numpy as np
import numpy.testing as testing


class TestOrbitals(unittest.TestCase):
    def setUp(self):
        self.grid = tdse.grid.ShGrid(100, 2, 20)
        self.atom = tdse.atom.Mg

        self.orbs = tdse.orbitals.Orbitals(self.atom, self.grid)
        self.orbs.init()
        self.data = self.orbs.asarray()

    def test_normalize(self):
        self.orbs.normalize()
        self.assertAlmostEqual(self.orbs.norm(), self.atom.countElectrons)

    def test_ort(self):
        self.orbs.normalize()
        self.orbs.ort()

        self.assertAlmostEqual(self.orbs.get_wf(0)*self.orbs.get_wf(1), 0.0)
        self.assertAlmostEqual(self.orbs.get_wf(0)*self.orbs.get_wf(2), 0.0)
        self.assertAlmostEqual(self.orbs.get_wf(0)*self.orbs.get_wf(3), 0.0)
