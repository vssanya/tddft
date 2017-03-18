import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tdse

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
orbs = atom.get_ground_state(grid=sh_grid, filename='./ar_gs_dr_0.02_dt_0.008.npy')

f = tdse.field.TwoColorPulseField(E0 = 0.0, alpha = 0.0)

r = np.linspace(dr,r_max,Nr)

def data_gen():
    t = 0.0
    while True:
        ws.prop(orbs, atom, f, t, dt)
        t += dt
        print(t)
        yield 1

fig, ax = plt.subplots()
lines = []
for ie in range(9):
    line, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2, label="n = {}".format(ie))
    lines.append(line)

ax.grid()
ax.set_ylim(1e-12, 1e3)
ax.set_yscale('log')

def run(data):
    arr = orbs.asarray()
    for ie in range(9):
        lines[ie].set_ydata(np.sum(np.abs(arr[ie])**2, axis=0))
    return lines,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=1, repeat=False)
plt.show()
