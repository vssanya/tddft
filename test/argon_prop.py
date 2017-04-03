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
orbs = tdse.orbitals.SOrbitals(atom, sh_grid)
orbs.load('./ar_gs_dr_0.02.npy')
orbs.normalize()

T = 2*np.pi / 5.7e-2
tp = T

f = tdse.field.SinField(
        E0=tdse.utils.I_to_E(2e15),
        alpha=0.0,
        tp=tp
        )

r = np.linspace(dr,r_max,Nr)

t = np.arange(0, tp, dt)

def data_gen():
    for i in range(t.size):
        ws.prop(orbs, atom, f, t[i], dt)
        print("t = {}, E = {}".format(t[i], f.E(t[i])))
        yield i

fig = plt.figure()
ax = plt.subplot(121)
ax_n = plt.subplot(122)

lines = []
for ie in range(7):
    line, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2, label="n = {}".format(ie))
    lines.append(line)

ax.grid()
ax.set_ylim(1e-12, 1e3)
ax.set_yscale('log')

n = np.zeros((t.size, 7))
az = np.zeros(t.size)
orbs.norm_ne(n[0,:], True)
print(n[0,:])

line_n, = ax_n.plot(t, az)
ax_n.set_ylim(-1, 1)
lines.append(line_n)

def run(data):
    i = data

    arr = orbs.asarray()
    for ie in range(7):
        lines[ie].set_ydata(np.sum(np.abs(arr[ie])**2, axis=0))

    orbs.norm_ne(n[i,:], True)
    az[i] = tdse.calc.az(orbs, atom, f, t[i])
    line_n.set_ydata(az)
    ax_n.set_xlim(0, t[i])
    print(n[i,:])
    return lines,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=1, repeat=False)
plt.show()
