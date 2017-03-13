import numpy as np
import matplotlib.pyplot as plt

from tdse import grid, wavefunc, orbitals, field, workspace, hydrogen, calc, utils, hartree_potential

dt = 0.0025
dr = 0.0125
r_max = 100
Nr=r_max/dr

g = grid.ShGrid(Nr=Nr, Nl=6, r_max=r_max)
sg = grid.SpGrid(Nr, 32, 2, r_max)

orbs = hydrogen.a_init(g)
r = np.linspace(dr,r_max,Nr)

arr = orbs.asarray()
arr[:] = 0.0
arr[0,0,:] = 1.0
#arr[0,1,0] = 1

uh0 = hartree_potential.l0(orbs)
uh1 = hartree_potential.l1(orbs)
uh2 = hartree_potential.l2(orbs)

ux = hartree_potential.lda(0, orbs, sg)

# plt.plot(r, uh0, label='uh0')
# plt.plot(r, uh1, label='uh1')
# plt.plot(r, uh2, label='uh2')
plt.plot(r, ux, label='ux')
# plt.plot(r, dr/r, '--', label='dr/r')
plt.plot(r, (3/np.pi)*(np.sum(np.abs(arr[0])**2, axis=0)/r**2)**1/3, '--', label='dr/r')
plt.grid()
plt.legend()
plt.show()
