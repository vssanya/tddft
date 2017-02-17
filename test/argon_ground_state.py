import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tdse import grid, wavefunc, orbitals, field, workspace, hydrogen, calc, utils

dt = 0.04
dr = 0.2
r_max = 200
Nr=r_max/dr

g = grid.SGrid(Nr=Nr, Nl=2, r_max=r_max)
ws = workspace.SOrbsWorkspace(dt=dt, grid=g)
orbs = hydrogen.a_init(g)
hydrogen.a_ort(orbs)
orbs.normalize()

r = np.linspace(dr,r_max,Nr) + 1.0

def data_gen():
    cnt = 0

    while cnt < 1000:
        ws.prop_img(orbs)
        n = np.sqrt(orbs.norm_ne())
        print(2/dt*(1-n)/(1+n))
        hydrogen.a_ort(orbs)
        orbs.normalize()
        yield 1

fig, ax = plt.subplots()
line1, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)
line2, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)
line3, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)
line4, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)
line5, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)
line6, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)
line7, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)
line8, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)
line9, = ax.plot(r, np.abs(orbs.asarray()[0,0])**2)

ax.grid()
# ax.set_xlim(0, 60)
ax.set_ylim(1e-10, 20)
ax.set_xscale('log')
ax.set_yscale('log')

def run(data):
    arr = orbs.asarray()
    line1.set_ydata(np.sum(np.abs(arr[0]*r)**2, axis=0))
    line2.set_ydata(np.sum(np.abs(arr[1]*r)**2, axis=0))
    line3.set_ydata(np.sum(np.abs(arr[2]*r)**2, axis=0))
    line4.set_ydata(np.sum(np.abs(arr[3]*r)**2, axis=0))
    line5.set_ydata(np.sum(np.abs(arr[4]*r)**2, axis=0))
    line6.set_ydata(np.sum(np.abs(arr[5]*r)**2, axis=0))
    line7.set_ydata(np.sum(np.abs(arr[6]*r)**2, axis=0))
    line8.set_ydata(np.sum(np.abs(arr[7]*r)**2, axis=0))
    line9.set_ydata(np.sum(np.abs(arr[8]*r)**2, axis=0))
    return (line1, line2),

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10, repeat=False)
plt.show()
