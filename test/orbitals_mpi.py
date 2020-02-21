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
        if self.rank == 0.0:
            self.orbs_base = tdse.orbitals.ShOrbitals(self.atom, self.grid)

    def test_collect(self):
        if self.rank == 0:
            data = np.zeros(self.orbs.shape, dtype=np.complex)
            data_coll = np.zeros(self.orbs.shape, dtype=np.complex)
            data[:] = np.random.random(data.shape)
        else:
            data = None
            data_coll = None

        self.orbs.load(data=data)
        self.orbs.collect(data_coll)

        if self.rank == 0:
            np.testing.assert_equal(data, data_coll)

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

    def test_orbs_rank(self):
        if self.rank == 0:
            data = np.zeros((4, 2, 100), dtype=np.complex)
            data[:] = np.random.random(data.shape)
            self.orbs_base.load(data=data)
        else:
            data = None

        self.orbs.load(data=data)

        orbs_rank = tdse.orbitals.ShOrbitals(self.atom, self.grid, self.comm, np.array([0, 0, 1, 1], dtype=np.intc))
        orbs_rank.load(data=data)

        v1 = self.orbs.norm()
        v2 = orbs_rank.norm()
        if self.rank == 0:
            v = self.orbs_base.norm()
            self.assertEqual(v, v1, 8)
            self.assertEqual(v, v2, 8)

        v1 = self.orbs.z()
        v2 = orbs_rank.z()
        if self.rank == 0:
            v = self.orbs_base.z()
            self.assertEqual(v, v1, 8)
            self.assertEqual(v, v2, 8)
