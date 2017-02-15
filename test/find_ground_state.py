import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tdse import grid, wavefunc, field, workspace, hydrogen, calc, utils

dt = 0.025
dr = 0.125
r_max = 60
g = grid.SGrid(Nr=r_max/dr, Nl=1, r_max=r_max)
wf = wavefunc.SWavefunc.random(g)
ws = workspace.SKnWorkspace(dt=dt, grid=g)

wf_hydrogen = hydrogen.ground_state(g)
r = np.linspace(0,10,Nr)

def data_gen():
    cnt = 0

    while cnt < 1000:
        ws.prop_img(wf)
        norm = wf.norm()
        wf.normalize()

        print("Energy =", (np.sqrt(norm) - 1)/dt)
        print("z =", wf.z())

        arr = wf.asarray()
        yield np.sum(np.abs(arr)**2, axis=0)

fig, ax = plt.subplots()

ax.plot(r, np.abs(wf_hydrogen.asarray()[0])**2)

line, = ax.plot(r, np.abs(wf.asarray()[0])**2)
ax.grid()
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)

def run(data):
    line.set_data(r, data)
    return line,

ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10, repeat=False)
plt.show()
