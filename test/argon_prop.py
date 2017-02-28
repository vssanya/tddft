import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tdse import grid, wavefunc, orbitals, field, atom, workspace, calc, utils

dt = 0.008
dr = 0.02
r_max = 200
Nr=r_max/dr

Ar = atom.Atom('Ar')
g = grid.SGrid(Nr=Nr, Nl=20, r_max=r_max)
ws = workspace.SOrbsWorkspace(grid=g, atom=Ar)
orbs = Ar.get_ground_state(grid=g, filename='./argon_ground_state.npy')

I0 = 1e14
freq = utils.length_to_freq(800, 'nm')
E0 = utils.I_to_E(I0)
tp = utils.t_fwhm(20, 'fs')
t0 = utils.t_shift(tp, I0, Imin=I0*1e-7)

f = field.TwoColorPulseField(
    E0 = E0,
    alpha = 0.0,
    freq = freq,
    phase = 0.0,
    tp = tp,
    t0 = 0.0
)

r = np.linspace(dr,r_max,Nr)


def data_gen():
    t = 0.0
    while True:
        ws.prop(orbs, f, t, dt)
        t += dt
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
