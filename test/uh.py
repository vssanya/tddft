import numpy as np
import matplotlib.pyplot as plt

from tdse import grid, wavefunc, orbitals, field, workspace, hydrogen, calc, utils, hartree_potential

dt = 0.0025
dr = 0.0125
r_max = 100
Nr=r_max/dr

g = grid.SGrid(Nr=Nr, Nl=6, r_max=r_max)
orbs = hydrogen.a_init(g)
r = np.linspace(dr,r_max,Nr)

arr = orbs.asarray()
# arr[:] = 0.0
# arr[0,0,0] = 1
# arr[0,1,0] = 1

uh0 = hartree_potential.l0(orbs)
uh1 = hartree_potential.l1(orbs)
uh2 = hartree_potential.l2(orbs)

plt.plot(r, uh0)
plt.plot(r, uh1)
plt.plot(r, uh2)
plt.plot(r, dr/r)
plt.show()
