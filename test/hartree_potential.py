import unittest

import tdse
import numpy as np


class TestHartreePotential(unittest.TestCase):
    def setUp(self):
        dr = 0.02
        r_max = 200
        Nr = int(r_max/dr)
        Nl = 1

        self.grid = tdse.grid.ShGrid(Nr, Nl, r_max)
        self.wf = tdse.wavefunc.ShWavefunc(self.grid)
        self.psi = self.wf.asarray()
        self.psi[0,:100] = 1.0
        self.psi[0,100:] = 0.0
        self.wf.normalize()
        self.uh = np.zeros(Nr)
        self.r = np.linspace(dr, r_max, Nr)

    def test_potential_wf(self):
        tdse.hartree_potential.wf_l0(self.wf, self.uh, 3)
        self.assertAlmostEqual((1/self.r/self.uh)[-1], 1.0)
