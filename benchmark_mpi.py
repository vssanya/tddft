import time
from mpi4py import MPI
import numpy as np

import tdse


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Nr = 10000

grid = tdse.grid.SGrid(Nr=Nr, Nl=60, r_max=200)

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


sp_grid = tdse.grid.SpGrid(Nr, 32, 1, 200)

n_wf = np.ndarray((Nr, 32))

uh = np.ndarray(Nr)
uxc = np.ndarray(Nr)

if rank == 0:
    n_orbs = np.ndarray((Nr, 32))
else:
    n_orbs = None

start = time.time()
tdse.mpi.orbitals_calc_uxc_l0(wf, sp_grid, n_wf, n_orbs, uxc, comm)
tdse.mpi.orbitals_calc_uh_l0(wf, uh, comm)
end = time.time()
if rank == 0:
    print("Calc u_lda time", end - start)
