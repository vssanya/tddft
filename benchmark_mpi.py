import time
from mpi4py import MPI

import numpy as np
from tdse import grid, wavefunc, orbitals, field, atom, workspace, calc, utils


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


g = grid.SGrid(Nr=20000, Nl=60, r_max=200)

if rank == 0:
    orbs = orbitals.SOrbitals(size, g)
else:
    orbs = None

wf = wavefunc.SWavefunc(g)

start = time.time()

if rank == 0:
    comm.Gatherv(wf.asarray(), orbs.asarray(), 0)
else:
    comm.Gatherv(wf.asarray(), None, 0)

end = time.time()

if rank == 0:
    print("Time send = ", end - start)
