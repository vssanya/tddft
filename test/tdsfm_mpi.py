import unittest

from mpi4py import MPI
import numpy as np
import numpy.testing as testing

import tdse


class TestOrbitals(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.grid = tdse.grid.ShGrid(100, 2, 20)
        self.atom = tdse.atom.Ne

        self.sp_grid = tdse.grid.SpGrid(10, 10, 1, 2)

        self.orbs = tdse.orbitals.ShOrbitals(self.atom, self.grid, self.comm)

        if self.rank == 0:
            data = np.zeros(self.orbs.shape, dtype=np.complex)
            data[:] = np.random.random(data.shape)
        else:
            data = None

        self.orbs.load(data=data)

    def test_collect(self):
        self.tdsfm = tdse.tdsfm.TDSFMOrbs(self.orbs, self.sp_grid, 10//self.grid.dr, tdse.GAUGE_VELOCITY, A_max = 0.0)

        if self.rank == 0:
            data_coll = np.zeros((self.orbs.shape[0], self.sp_grid.Nc, self.sp_grid.Nr), dtype=np.complex)
            data_base = np.random.random(data_coll.shape)
            data_coll[:] = data_base
        else:
            data_coll = None

        self.tdsfm.collect(data_coll)

        if self.rank == 0:
            np.testing.assert_equal(data_coll, 0.0)
