import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tdse import grid, wavefunc, orbitals, field, workspace, hydrogen, calc, utils

dt = 0.025
dr = 0.125
r_max = 100
Nr=r_max/dr

g = grid.SGrid(Nr=Nr, Nl=2, r_max=r_max)
ws = workspace.SKnWorkspace(dt=dt, grid=g)
orbs = hydrogen.a_init(g)
hydrogen.a_ort(orbs)
orbs.normalize()

r = np.linspace(dr,r_max,Nr)

def data_gen():
    cnt = 0

    while cnt < 1000:
        ws.orbs_prop_img(orbs)
        hydrogen.a_ort(orbs)
        orbs.normalize()
        yield 1

fig, ax = plt.subplots()
line1, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)
line2, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)
line3, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)

ax.grid()
ax.set_xlim(0, 60)
ax.set_ylim(0, 0.5)

def run(data):
    arr = orbs.asarray()
    line1.set_ydata(np.sum(np.abs(arr[0])**2, axis=0))
    line2.set_ydata(np.sum(np.abs(arr[1])**2, axis=0))
    line3.set_ydata(np.sum(np.abs(arr[2])**2, axis=0))
    return (line1, line2),

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10, repeat=False)
plt.show()
