import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tdse import grid, wavefunc, field, workspace, hydrogen, calc, utils

def calc_az_t():
    freq = utils.length_to_freq(800, 'nm')
    T = 2*np.pi/freq

    E0 = 0.0377
    #E0 = 0.0

    tp = utils.t_fwhm(2.05, 'fs')
    t0 = 1.5*T

    f = field.TwoColorPulseField(
        E0 = E0,
        alpha = 0.0,
        freq = freq,
        phase = 0.0,
        tp = tp,
        t0 = t0
    )

    dt = 0.025
    dr = 0.125
    r_max = 60

    r = np.linspace(dr, r_max, r_max/dr)
    g = grid.SGrid(Nr=r_max/dr, Nl=20, r_max=r_max)
    wf = hydrogen.ground_state(g)
    ws = workspace.SKnWorkspace(dt=dt, grid=g)

    t = np.arange(0, 2*t0, dt)
    az    = np.zeros(t.size)
    z    = np.zeros(t.size)

    def data_gen():
        for it in range(t.size):
            ws.prop(wf, f, t[it])
            yield it, calc.az(wf, f, t[it])

    fig = plt.figure()

    ax1 = plt.subplot(221)
    ax1.grid()
    ax1.set_xlim(t[0], t[-1])

    ax2 = plt.subplot(222)
    ax2.set_xlim(r[0], r[-1])
    ax2.set_ylim(1e-12,1)
    ax2.set_yscale('log')

    ax3 = plt.subplot(223)
    ax3.set_xlim(t[0], t[-1])

    ax4 = plt.subplot(224)

    line1,  = ax1.plot(t, az)
    line3, = ax2.plot(r, np.sum(np.abs(wf.asarray()), axis=0))

    E = np.zeros(t.size)
    for i in range(t.size):
        E[i] = f.E(t[i])
    line4, = ax3.plot(t, E)
    line5, = ax3.plot([0,], [E[0],], '.')

    line6, = ax4.plot(t, z)

    def run(data):
        it = data[0]
        az[it] = data[1]
        z[it] = wf.z()

        ax1.set_xlim(t[0], t[it])
        ax4.set_xlim(t[0], t[it])

        it = it+1
        ax1.set_ylim(np.min(az[0:it]), np.max(az[0:it]))
        ax4.set_ylim(np.min(z[0:it]), np.max(z[0:it]))

        line1.set_ydata(az)
        line6.set_ydata(z)

        line3.set_ydata(np.sum(np.abs(wf.asarray()), axis=0))
        line5.set_data([t[it-1],],[E[it],])

        return (line1, line2, line3, line5),

    ani = animation.FuncAnimation(fig, run, data_gen, blit=False,
            interval=10, repeat=False)
    plt.show()
    return az

if __name__ == "__main__":
    calc_az_t()
