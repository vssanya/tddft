import unittest

import tdse
import numpy as np
import numpy.testing as testing


class TestOrbitals(unittest.TestCase):
    def setUp(self):
        self.grid = tdse.grid.ShGrid(100, 4, 20)
        self.atom = tdse.atom.Ar

        self.orbs = tdse.orbitals.ShOrbitals(self.atom, self.grid)
        self.orbs.init()
        self.data = self.orbs.asarray()

    def test_init(self):
        psi = self.orbs.asarray()
        print(psi)
        testing.assert_array_equal(psi[0, 1:], 0.0)
        testing.assert_array_equal(psi[1, 1:], 0.0)
        testing.assert_array_equal(psi[2, 1:], 0.0)
        testing.assert_array_equal(psi[0:2, 0] != 0.0, True)

    def test_normalize(self):
        self.orbs.normalize()
        self.assertAlmostEqual(self.orbs.norm(), self.atom.countElectrons)

    def test_ort(self):
        self.orbs.normalize()
        self.orbs.ort()

        self.assertAlmostEqual(self.orbs.get_wf(0)*self.orbs.get_wf(1), 0.0)
        self.assertAlmostEqual(self.orbs.get_wf(0)*self.orbs.get_wf(2), 0.0)
        self.assertAlmostEqual(self.orbs.get_wf(0)*self.orbs.get_wf(3), 0.0)
