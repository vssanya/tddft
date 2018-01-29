import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tdse


def calc_wf_az_t():
    freq=tdse.utils.length_to_freq(800, 'nm')
    f = tdse.field.CarField(
            E=tdse.utils.I_to_E(1e14),
            freq=freq,
            phase=0
        )*tdse.field.SinEnvField(2*np.pi/freq)

    dt = 0.025
    dr = 0.125

    r_max = 50
    Nl = 8

    t = f.get_t(dt, 0)

    atom = tdse.atom.H
    r = np.linspace(dr, r_max, r_max/dr)
    g = tdse.grid.ShGrid(Nr=r_max/dr, Nl=Nl, r_max=r_max)

    uabs = tdse.abs_pot.UabsMultiHump(2.5, 10)

    ws = tdse.workspace.SKnWorkspace(grid=g, uabs=uabs)
    ws_a = tdse.workspace.SKnAWorkspace(g, uabs)

    wf = tdse.ground_state.wf(atom, g, ws, dt*4, Nt=500, n=1, l=0, m=0)
    wf_a = tdse.ground_state.wf(atom, g, ws_a, dt*4, Nt=500, n=1, l=0, m=0)

    data = wf.asarray()

    az    = np.zeros(t.size)
    z    = np.zeros(t.size)

    az_a    = np.zeros(t.size)
    z_a    = np.zeros(t.size)

    def data_gen():
        for i in range(t.size):
            ws.prop(wf, atom, f, t[i], dt)
            az[i] = tdse.calc.az(wf, atom, f, t[i])
            #az[i] = wf.z()

            ws_a.prop(wf_a, atom, f, t[i], dt)
            az_a[i] = tdse.calc.az(wf_a, atom, f, t[i])
            #az_a[i] = wf_a.z()

            print("az = ", az[i])
            print("az_a = ", az_a[i])

            if (i+1) % 10 == 0:
                yield i

    fig = plt.figure()

    ax1 = plt.subplot(221)
    ax1.grid()
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(-1e-6, 1e-6)

    ax2 = plt.subplot(222)
    #ax2.set_xlim(r[0], r[-1])
    ax2.set_xlim(r[0], r_max)
    ax2.set_ylim(1e-60,1e1)
    ax2.set_yscale('log')
    #ax2.set_xscale('log')

    ax3 = plt.subplot(223)
    ax3.set_xlim(t[0], t[-1])

    ax4 = plt.subplot(224)
    ax4.set_xlim(r[0], r_max)
    #ax4.set_xscale('log')
    ax4.set_ylim(1e-6, 100)
    ax4.set_yscale('log')

    line1,  = ax1.plot(t, az, label="az")
    line1_a,  = ax1.plot(t, az_a, label="az_a")
    ax1.legend()

    line3, = ax2.plot(r, np.sum(np.abs(wf.asarray()), axis=0), label='Num')
    line3_a, = ax2.plot(r, np.sum(np.abs(wf_a.asarray()), axis=0), label='Num A')
    ax2.legend()

    E = np.zeros(t.size)
    for i in range(t.size):
        E[i] = f.E(t[i])
    line4, = ax3.plot(t, E)
    line5, = ax3.plot([0,], [E[0],], '.')

    ax1.plot(t, np.abs(E))


    def run(i):
        ax1.set_xlim(t[0], t[i])

        line1.set_ydata(az)
        line1_a.set_ydata(az_a)
        ax1.set_ylim(np.nanmin((az[0:i], az_a[0:i])), np.nanmax((az[0:i], az_a[0:i])))

        line3.set_ydata(np.sum(np.abs(wf.asarray())**2, axis=0))
        line3_a.set_ydata(np.sum(np.abs(wf_a.asarray())**2, axis=0))

        return (line1, line1_a, line3),

    ani = animation.FuncAnimation(fig, run, data_gen, blit=False,
            interval=10, repeat=False)
    plt.legend()
    plt.show()
    return az

if __name__ == "__main__":
    calc_wf_az_t()
