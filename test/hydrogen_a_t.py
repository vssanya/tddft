import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tdse


def calc_wf_az_t():
    freq = tdse.utils.length_to_freq(800, 'nm')
    T = 2*np.pi/freq

    I = 1e14
    E0 = tdse.utils.I_to_E(I)

    tp = tdse.utils.t_fwhm(2.05, 'fs')
    t0 = 1.5*T

    f = tdse.field.TwoColorPulseField(E0=E0, freq=freq, alpha=0.0, tp=tp, t0=t0)

    dt = 0.025
    dr = 0.125
    r_max = 100

    a = tdse.atom.Atom('H')
    r = np.linspace(dr, r_max, r_max/dr)
    g = tdse.grid.ShGrid(Nr=r_max/dr, Nl=32, r_max=r_max)

    orbs = tdse.orbitals.SOrbitals(a, g)
    orbs.init()
    orbs.normalize()
    wf = orbs.get_wf(0)
    ws = tdse.workspace.SKnWorkspace(grid=g, num_threads=2)

    for i in range(1000):
        ws.prop_img(wf, a, dt)
        wf.normalize()

    arr1 = np.array(wf.asarray())
    print(arr1)
    ws.prop(wf, a, f, 0.0, dt)
    arr2 = np.array(wf.asarray())
    plt.plot(np.abs(arr2)[0] - np.abs(arr1)[0])
    plt.plot(np.abs(arr2)[1] - np.abs(arr1)[1])
    plt.show()

    t = np.arange(0, 2*t0, dt)
    az    = np.zeros(t.size)
    prob = np.zeros(t.size)
    z    = np.zeros(t.size)

    def data_gen():
        for i in range(t.size):
            print(f.E(t[i]))
            yield i, tdse.calc.az(wf, a, f, t[i])
            ws.prop(wf, a, f, t[i], dt)

    fig = plt.figure()

    ax1 = plt.subplot(221)
    ax1.grid()
    ax1.set_xlim(t[0], t[-1])

    ax2 = plt.subplot(222)
    #ax2.set_xlim(r[0], r[-1])
    ax2.set_xlim(0, 200)
    ax2.set_ylim(1e-16,1e2)
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
        prob[it] = wf.norm()#tdse.calc.ionization_prob(orbs)
        #z[it] = wf.z()

        ax1.set_xlim(t[0], t[it])
        ax4.set_xlim(t[0], t[it])

        it = it+1
        #ax1.set_ylim(np.min(az[0:it]), np.max(az[0:it]))
        ax1.set_ylim(np.min(prob[0:it]), np.max(prob[0:it]))
        ax4.set_ylim(np.min(az[0:it]), np.max(az[0:it]))

        line6.set_ydata(az)
        line1.set_ydata(prob)
        #line6.set_ydata(z)

        line3.set_ydata(np.sum(np.abs(wf.asarray()), axis=0))
        line5.set_data([t[it],],[E[it],])

        return (line1, line3, line5, line6),

    ani = animation.FuncAnimation(fig, run, data_gen, blit=False,
            interval=10, repeat=False)
    plt.show()
    return az

if __name__ == "__main__":
    calc_wf_az_t()
