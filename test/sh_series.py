import unittest

import tdse
import numpy as np


class TestShSeries(unittest.TestCase):
    def setUp(self):
        self.sp_grid = tdse.grid.SpGrid(1, 32*4, 1, 1)
        self.sp_grid_2d = tdse.grid.SpGrid2d(1, 32*4, 1)
        self.sh_grid = tdse.grid.ShGrid(1, 32, 1)
        self.ylm = tdse.sphere_harmonics.YlmCache(32, self.sp_grid)

    def test_convert(self):
        sh_wf = tdse.wavefunc.ShWavefunc(self.sh_grid, 0)
        data = sh_wf.asarray()
        data[:] = 1.0

        sp_wf = tdse.wavefunc.SpWavefunc(self.sp_grid_2d, 0)
        tdse.wavefunc.convert_sh_to_sp(sh_wf, sp_wf, self.ylm, 0)

        sh_wf_convert = tdse.wavefunc.ShWavefunc(self.sh_grid, 0)
        tdse.wavefunc.convert_sp_to_sh(sp_wf, sh_wf_convert, self.ylm, 0)

        np.testing.assert_almost_equal(sh_wf.asarray(), sh_wf_convert.asarray(), decimal=2)
