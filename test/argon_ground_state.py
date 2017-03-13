import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tdse import grid, wavefunc, orbitals, field, atom, workspace, calc, utils

dt = 0.008
dr = 0.02
r_max = 200
Nr=r_max/dr

Ar = atom.Atom('Ar')
g = grid.ShGrid(Nr=Nr, Nl=2, r_max=r_max)
ws = workspace.SOrbsWorkspace(grid=g, atom=Ar)
orbs = Ar.get_init_orbs(g)

r = np.linspace(dr,r_max,Nr) + 1.0

def data_gen():
    while True:
        #for i in range(10):
        Ar.ort(orbs)
        orbs.normalize()
        ws.prop_img(orbs, dt)

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

np.save('argon_ground_state.npy', orbs.asarray())
