import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tdse

dt = 0.032
dr = 0.02
r_max = 200
Nr=r_max/dr
Nl=32

atom = tdse.atom.Atom('Ar')
sh_grid = tdse.grid.ShGrid(Nr=Nr, Nl=Nl, r_max=r_max)
sp_grid = tdse.grid.SpGrid(Nr=Nr, Nc=Nl, Np=1, r_max=r_max)
ylm_cache = tdse.sphere_harmonics.YlmCache(Nl, sp_grid)
ws = tdse.workspace.SOrbsWorkspace(sh_grid, sp_grid, ylm_cache)
orbs = atom.get_ground_state(grid=sh_grid, filename='./ar_gs_dr_0.02.npy')
orbs.normalize()

f = tdse.field.TwoColorPulseField(E0 = 0.0, alpha = 0.0)

r = np.linspace(dr,r_max,Nr)

t = np.linspace(0, dt*10000, 10000)

def data_gen():
    for i in range(t.size):
        ws.prop(orbs, atom, f, t[i], dt)
        print(t[i])
        yield i

fig = plt.figure()
ax = plt.subplot(121)
ax_n = plt.subplot(122)

lines = []
for ie in range(9):
    line, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2, label="n = {}".format(ie))
    lines.append(line)

ax.grid()
ax.set_ylim(1e-12, 1e3)
ax.set_yscale('log')

n = np.zeros((t.size, 9))
line_n, = ax_n.plot(t, n[:,0])
ax_n.set_ylim(1-1e-5, 1+1e-5)
ax_n.set_yscale('log')
lines.append(line_n)

def run(data):
    i = data

    arr = orbs.asarray()
    for ie in range(9):
        lines[ie].set_ydata(np.sum(np.abs(arr[ie])**2, axis=0))

    orbs.norm_ne(n[i,:], True)
    line_n.set_ydata(n[:,0])
    ax_n.set_xlim(0, t[i])
    print(n[i,:])
    return lines,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=1, repeat=False)
plt.show()
