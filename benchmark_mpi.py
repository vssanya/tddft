import time
from mpi4py import MPI
import numpy as np

import tdse


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


grid = tdse.grid.SGrid(Nr=10000, Nl=60, r_max=200)

if rank == 0:
    orbs = tdse.orbitals.SOrbitals(size, grid)
else:
    orbs = None

wf = tdse.wavefunc.SWavefunc(grid)
ws = tdse.workspace.SOrbsWorkspace(grid, tdse.atom.Atom('H'))


for i in range(20):
    start = time.time()
    if rank == 0:
        comm.Gatherv(wf.asarray(), orbs.asarray(), 0)
    else:
        comm.Gatherv(wf.asarray(), None, 0)
    end = time.time()

    if rank == 0:
        print("Time send = ", end - start)


sp_grid = tdse.grid.SpGrid(10000, 32, 1, 200)

n_wf = np.ndarray((10000, 32))
if rank == 0:
    n_orbs = np.ndarray((10000, 32))
else:
    n_orbs = None

start = time.time()
tdse.mpi.orbitals_calc_n_sp(wf, sp_grid, n_wf, n_orbs, comm)
if rank == 0:
    u_lda = tdse.hartree_potential.lda_n(0, n_orbs, sp_grid)

end = time.time()
if rank == 0:
    print("Calc u_lda time", end - start)
