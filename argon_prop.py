import matplotlib.pyplot as plt
import time
from mpi4py import MPI
import numpy as np

import tdse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

I = 1e15
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
ws = tdse.workspace.SOrbsWorkspace(sh_grid, sp_grid, atom, ylm_cache)

orbs = atom.get_ground_state(grid=sh_grid, filename='./argon_ground_state_new.npy', comm=comm)
orbsl = atom.get_ground_state(grid=sh_grid, filename='./argon_ground_state_new.npy')

orbs.normalize()
orbsl.normalize()

freq = tdse.utils.length_to_freq(800, 'nm')
tp = 20*(2*np.pi/freq)

f = tdse.field.SinField(
    E0 = 0.0,
    alpha = 0.0,
    freq = freq,
    phase = 0.0,
    tp = tp,
    t0 = 0.0
)

r = np.linspace(dr,r_max,Nr)
t = np.arange(0, tp, dt)

for i in range(10):
    ws.prop(orbs, f, t[i], dt)
    ws.prop(orbsl, f, t[i], dt)

    prob = tdse.calc.ionization_prob(orbs)
    if rank == 0:
        print(prob)
        print(tdse.calc.ionization_prob(orbsl))
