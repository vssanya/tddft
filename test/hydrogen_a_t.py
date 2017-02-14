import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tdse import grid, wavefunc, field, workspace, hydrogen, calc, utils

def calc_az_t():
    freq = utils.length_to_freq(800, 'nm')
    T = 2*np.pi/freq

    E0 = utils.I_to_E(1e14)

    tp = utils.t_fwhm(9.33, 'fs')
    t0 = 8*T

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
    r_max = 100

    g = grid.SGrid(Nr=r_max/dr, Nl=80, r_max=r_max)
    wf = hydrogen.ground_state(g)
    ws = workspace.SKnWorkspace(dt=dt, grid=g)

    t = np.arange(0, 2*t0, dt)
    az    = np.zeros(t.size)
    az_lf = np.zeros(t.size)

    def data_gen():
        for it in range(t.size):
            ws.prop(wf, f, t[it])
            yield it, calc.az(wf, f, t[it]), calc.az_lf(wf, f, t[it])


    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlim(t[0], t[-1])

    line1, = ax.plot([], [], lw=2, color='b')
    line2, = ax.plot([], [], lw=2, color='r')

    def run(data):
        it = data[0]
        az[it] = data[1]
        az_lf[it] = data[2]

        it = it+1
        ax.set_ylim(np.min(az[0:it]), np.max(az[0:it]))
        ax.set_xlim(t[0], t[it-1])

        line1.set_data(t[0:it], az[0:it])
        line2.set_data(t[0:it], az_lf[0:it])

        return (line1, line2),

    ani = animation.FuncAnimation(fig, run, data_gen, blit=False,
            interval=10, repeat=False)
    plt.show()
    return az, az_lf

if __name__ == "__main__":
    calc_az_t()
