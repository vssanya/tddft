import matplotlib.pyplot as plt
import time
from mpi4py import MPI
import numpy as np

import tdse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

I = 2e14
E = tdse.utils.I_to_E(I)

dt = 0.008
dr = 0.02
r_max = 200
Nr=r_max/dr
Nl=32

atom = tdse.atom.Atom('Ar')
sh_grid = tdse.grid.ShGrid(Nr=Nr, Nl=Nl, r_max=r_max)
sp_grid = tdse.grid.SpGrid(Nr=Nr, Nc=Nl, Np=1, r_max=r_max)
ylm_cache = tdse.sphere_harmonics.YlmCache(Nl, sp_grid)
ws = tdse.workspace.SOrbsWorkspace(sh_grid, sp_grid, ylm_cache)

orbs = atom.get_ground_state(grid=sh_grid, filename='./ar_gs_dr_0.02_dt_0.008.npy', comm=comm)

orbs.normalize()

freq = tdse.utils.length_to_freq(800, 'nm')
tp = 20*(2*np.pi/freq)

f = tdse.field.SinField(
    E0 = E,
    alpha = 0.0,
    freq = freq,
    phase = 0.0,
    tp = tp,
    t0 = 0.0
)

t = np.arange(0, tp, dt)
if rank == 0:
    print(t.size)

    orbs_n = np.zeros((t.size, 9))
    az = np.zeros(t.size)
else:
    orbs_n = None
    az = None

for i in range(t.size):
    start = time.time()
    ws.prop(orbs, atom, f, t[i], dt)

    orbs.norm_ne(orbs_n[i,:], True)
    az_i = tdse.calc.az(orbs, atom, f, t[i])
    end = time.time()

    if rank == 0:
        print("dt = ", end - start)
        az[i] = az_i
        if i % int(t.size*0.01) == 0:
            np.save('az_ar.npy', az)
            np.save('orbs_n_ar.npy', orbs_n)
            print("Status: ", i // int(t.size*0.01) * 100)

if rank == 0:
    np.save('az_ar.npy', az)
    np.save('orbs_n_ar.npy', orbs_n)
