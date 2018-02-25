import unittest

import tdse


class TestTdsfm(unittest.TestCase):
    def setUp(self):
        self.r_grid = tdse.grid.ShGrid(10,10,10)
        self.k_grid = tdse.grid.SpGrid(10, 4, 1, 4)

    def test_calc(self):
        tdsfm = tdse.tdsfm.TDSFM(self.k_grid, self.r_grid, 10)
        wf = tdse.wavefunc.ShWavefunc(self.r_grid)
        field = tdse.field.CarField(1,1,1)

        tdsfm.calc(field, wf, 0, 0.1)
