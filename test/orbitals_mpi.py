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

        self.orbs = tdse.orbitals.ShOrbitals(self.atom, self.grid, self.comm)
        self.data = self.orbs.asarray()

    def test_load_low(self):
        if self.rank == 0:
            data = np.zeros((4, 2, 10), dtype=np.complex)
            data[:] = np.random.random(data.shape)
        else:
            data = None

        self.orbs.load(data=data)

        data = self.comm.bcast(data, root=0)

        np.testing.assert_equal(data[self.rank], self.orbs.asarray()[0,:,:10])
        np.testing.assert_equal(self.orbs.asarray()[0,:,10:], 0.0)

    def test_load_high(self):
        if self.rank == 0:
            data = np.zeros((4, 2, 200), dtype=np.complex)
            data[:] = np.random.random(data.shape)
        else:
            data = None

        self.orbs.load(data=data)

        data = self.comm.bcast(data, root=0)

        np.testing.assert_equal(data[self.rank][:,:100], self.orbs.asarray()[0])
