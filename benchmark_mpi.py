import time
from mpi4py import MPI
import numpy as np

import tdse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Nr = 10000

grid = tdse.grid.ShGrid(Nr=Nr, Nl=60, r_max=200)
sp_grid = tdse.grid.SpGrid(Nr=Nr, Nc=32, Np=1, r_max=200)
ylm_cache = tdse.sphere_harmonics.YlmCache(60, sp_grid)

orbs = tdse.orbitals.Orbitals(size, grid, comm=comm)
#orbs = tdse.orbitals.Orbitals(size, grid)
atom = tdse.atom.Atom('H')

wf = tdse.wavefunc.ShWavefunc(grid)
ws = tdse.workspace.SOrbsWorkspace(grid, sp_grid, ylm_cache)

n_wf = np.ndarray((Nr, 32))

uh = np.ndarray(Nr)
uxc = np.ndarray(Nr)

if rank == 0:
    countOrbs = np.ndarray((Nr, 32))
else:
    countOrbs = None

f = tdse.field.TwoColorPulseField()
ws.prop(orbs, atom, f, 0.0, 0.1)
