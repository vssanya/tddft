import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tdse

dt = 0.002
dr = 0.02
r_max = 200
Nr=r_max/dr
Nl = 2

Ar = tdse.atom.Atom('Ar')
g = tdse.grid.ShGrid(Nr=Nr, Nl=Nl, r_max=r_max)
sp_grid = tdse.grid.SpGrid(Nr=Nr, Nc=32, Np=1, r_max=r_max)
ylm_cache = tdse.sphere_harmonics.YlmCache(Nl, sp_grid)
ws = tdse.workspace.SOrbsWorkspace(g, sp_grid, ylm_cache)
orbs = Ar.get_ground_state(grid=g, filename='./ar_gs_dr_0.02_dt_0.008.npy')

r = np.linspace(dr,r_max,Nr) + 1.0

def data_gen():
    while True:
        #for i in range(10):
        Ar.ort(orbs)
        orbs.normalize()
        ws.prop_img(orbs, Ar, dt)

        n = np.sqrt(orbs.norm_ne())
        print(2/dt*(1-n)/(1+n))

        if np.max(np.abs(1 - n)) < 1e-7:
            break

        yield 1

fig, ax = plt.subplots()
lines = []
for ie in range(9):
    line, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2, label="n = {}".format(ie))
    lines.append(line)

ax.grid()
ax.set_ylim(1e-12, 100)
ax.set_yscale('log')

def run(data):
    arr = orbs.asarray()
    for ie in range(9):
        lines[ie].set_ydata(np.sum(np.abs(arr[ie]/r)**2, axis=0))
    return lines,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=1, repeat=False)
plt.show()

np.save('ar_gs_dr_0.02_not_lda.npy', orbs.asarray())
