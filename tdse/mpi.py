from mpi4py import MPI
import numpy as np

from . import wavefunc, grid


def orbitals_calc_n_sp(wf: wavefunc.SWavefunc, grid: grid.SpGrid, n_wf: np.ndarray, n_orbs: np.ndarray, comm: MPI.Comm, root: int = 0):
    wf.n_sp(grid, n_wf)
    comm.Reduce(n_wf, n_orbs, op=MPI.SUM, root=root)
