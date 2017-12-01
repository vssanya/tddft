import unittest

import tdse
import numpy as np
import numpy.testing

import scipy.special


class TestYlmCache(unittest.TestCase):
    def setUp(self):
        grid = tdse.grid.SpGrid(100, 128, 1, 4)
        self.ylm = tdse.sphere_harmonics.YlmCache(64, grid)

    def test_ylm(self):
        c = np.linspace(-1.0, 1.0, 256)
        res = np.zeros(c.size)

        for l in range(16):
            for i in range(c.size):
                res[i] = self.ylm(l, 0, c[i])

            numpy.testing.assert_array_almost_equal(res, np.real(scipy.special.sph_harm(0, l, 0, np.arccos(c))))
