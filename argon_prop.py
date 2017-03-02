import time
from mpi4py import MPI
import numpy as np

from tdse import grid, wavefunc, orbitals, field, atom, workspace, calc, utils


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

I = np.linspace(1e14, 5e14, size)
E = utils.I_to_E(I)

dt = 0.008
dr = 0.02
r_max = 200
Nr=r_max/dr

Ar = atom.Atom('Ar')
g = grid.SGrid(Nr=Nr, Nl=20, r_max=r_max)
ws = workspace.SOrbsWorkspace(grid=g, atom=Ar)
orbs = Ar.get_ground_state(grid=g, filename='./argon_ground_state.npy')
orbs.normalize()

freq = utils.length_to_freq(800, 'nm')
tp = 20*(2*np.pi/freq)

f = field.SinField(
    E0 = E[rank],
    alpha = 0.0,
    freq = freq,
    phase = 0.0,
    tp = tp,
    t0 = 0.0
)

r = np.linspace(dr,r_max,Nr)
t = np.arange(0, tp, dt)

start = time.time()
for i in range(10):
    ws.prop(orbs, f, t[i], dt)
print("Time:", time.time() - start)
print(t.size)

prob = calc.ionization_prob(orbs)

prob = comm.gather(prob, root=0)
if rank == 0:
    res = np.array(prob)
    np.savetxt('argon_prob_res.txt', res)
