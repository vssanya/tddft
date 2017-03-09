from mpi4py import MPI
import numpy as np

from . import wavefunc, grid, hartree_potential


def orbitals_calc_n_sp(wf: wavefunc.SWavefunc, grid: grid.SpGrid, n_wf: np.ndarray, n_orbs: np.ndarray, comm: MPI.Comm, root: int = 0):
    wf.n_sp(grid, n_wf)
    comm.Reduce(n_wf, n_orbs, op=MPI.SUM, root=root)


def orbitals_calc_uxc_l0(wf: wavefunc.SWavefunc, grid: grid.SpGrid, n_wf: np.ndarray, n_orbs: np.ndarray, uxc: np.ndarray, comm: MPI.Comm, root: int = 0):
    orbitals_calc_n_sp(wf, grid, n_wf, n_orbs, comm)
    if comm.Get_rank() == root:
        hartree_potential.lda_n(0, n_orbs, grid, uxc=uxc)
    comm.Bcast(uxc, root=root)


def orbitals_calc_uh_l0(wf: wavefunc.SWavefunc, uh: np.ndarray, comm: MPI.Comm, root: int = 0):
    hartree_potential.wf_l0(wf, uh)
    comm.Allreduce(MPI.IN_PLACE, uh, op=MPI.SUM)
