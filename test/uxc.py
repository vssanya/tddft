import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tdse

dt = 0.008
dr = 0.02
r_max = 120
Nr=r_max/dr
Nl=6

atom = tdse.atom.Ar
sh_grid = tdse.grid.ShGrid(Nr=Nr, Nl=Nl, r_max=r_max)
sp_grid = tdse.grid.SpGrid(Nr=Nr, Nc=33, Np=1, r_max=r_max)
ylm_cache = tdse.sphere_harmonics.YlmCache(Nl, sp_grid)

r = np.linspace(0, 120, Nr)

orbs = tdse.orbitals.SOrbitals(atom, sh_grid)
orbs.load('ar_r_120_lb.npy')

n_sp = orbs.n_sp(sp_grid, ylm_cache)
n = orbs.n_l0()

plt.plot(n)
plt.show()

uxc_l0 = tdse.hartree_potential.UXC_LB.calc_l0(0, orbs, sp_grid, ylm_cache, n=n_sp)
uxc_l = tdse.hartree_potential.UXC_LDA.calc_l(0, orbs, sp_grid, ylm_cache, n=n_sp)

plt.plot(r, uxc_l0, label='l0')
plt.plot(r, uxc_l, label='l')

plt.show()
