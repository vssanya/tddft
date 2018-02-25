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
uabs = tdse.abs_pot.UabsMultiHump(20*dr, r_max/8)
uabs = tdse.abs_pot.UabsZero()
ws = tdse.workspace.SOrbsWorkspace(sh_grid, sp_grid, uabs, ylm_cache, Uh_lmax=1, Uxc_lmax=3)
orbs = tdse.orbitals.Orbitals(atom, sh_grid)
orbs.load('ar_r_120_lb.npy')
#orbs.normalize()

T = 2*np.pi / 5.7e-2
tp = 20*T

f = tdse.field.TwoColorSinField(
        E0=tdse.utils.I_to_E(2e14),
        alpha=0.0,
        tp=tp
        )

r = np.linspace(dr,r_max,Nr)

t = np.arange(0, tp, dt)

def data_gen():
    for i in range(t.size):
        #if i % 20 == 0:
        print("t = {}, E = {}".format(t[i], f.E(t[i])))
        yield i
        ws.prop(orbs, atom, f, t[i], dt)

fig = plt.figure()
ax = plt.subplot(121)
ax_n = plt.subplot(122)

lines = []
for ie in range(atom.countOrbs):
    line, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2, label="n = {}".format(ie))
    lines.append(line)

ax.grid()
ax.set_ylim(1e-20, 1e3)
ax.set_yscale('log')

n = np.zeros((t.size, atom.countOrbs))
az = np.zeros(t.size)
z = np.zeros(t.size)
orbs.norm_ne(n[0,:], True)
print(n[0,:])

line_n, = ax_n.plot(t, az, label="az")
ax_n.set_ylim(-1e-6, 1e-6)
lines.append(line_n)

line_az, = ax_n.plot(t[1:-1], np.diff(z,2)/dt**2, label="z")
lines.append(line_az)

def run(data):
    i = data

    arr = orbs.asarray()
    for ie in range(atom.countOrbs):
        lines[ie].set_ydata(np.sum(np.abs(arr[ie])**2, axis=0))

    orbs.norm_ne(n[i,:], True)
    az[i] = tdse.calc.az(orbs, atom, f, t[i])
    z[i] = orbs.z()
    print("az = ", az[i])
    line_n.set_ydata(az)
    line_az.set_ydata(np.diff(z,2)/dt**2)
    ax_n.set_xlim(0, t[i])
    ax_n.set_ylim(np.min(az[0:i+1]), np.max(az[0:i+1]))
    print(n[i,:])
    return lines,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=1, repeat=False)
plt.legend()
plt.show()
