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

    #f = tdse.field.TwoColorPulseField(E0=0.0377, freq=freq, alpha=0.0, tp=tp, t0=t0)

    tp = T
    f = tdse.field.SinField(
            E0=tdse.utils.I_to_E(1e14),
            alpha=0.0,
            tp=tp
            )

    dt = 0.025
    dr = 0.125
    r_max = 100
    Nl = 2

    atom = tdse.atom.Atom('H')
    r = np.linspace(dr, r_max, r_max/dr)
    g = tdse.grid.ShGrid(Nr=r_max/dr, Nl=Nl, r_max=r_max)

    orbs = tdse.orbitals.SOrbitals(atom, g)
    orbs.init()
    orbs.normalize()
    wf = orbs.get_wf(0)
    data = wf.asarray()
    uabs = tdse.abs_pot.UabsMultiHump(dr*10, r_max/8)
    #uabs = tdse.abs_pot.UabsZero()
    ws = tdse.workspace.SKnWorkspace(grid=g, uabs=uabs, num_threads=2)

    #eigen = tdse.workspace.Eigen(g)
    #eigen.calc(atom)
    #eigen.save("./eigen.npy")
    #eigen.load("./eigen.npy")

    #gps_ws = tdse.workspace.GPSWorkspace(g, atom, dt, 2)
    #gps_ws.calc_s(eigen)

    #psi = wf.asarray()
    #psi[:] = 0.0
    #psi[0,:] = -eigen.get_evec()[0,:,0]/np.sqrt(dr)

    for i in range(10000):
        ws.prop_img(wf, atom, dt)
        wf.normalize()

    # data[:] = 0.0
    # data[1,:] = np.exp(-20*(r-25)**2)*np.exp(20j*r)

    #wf.asarray()[0,:] = np.exp(-(r-50)**2)*np.exp(10j*r)
    #dt = 0.004

    t = np.arange(0, tp, dt)
    az    = np.zeros(t.size)
    prob = np.zeros(t.size)
    z    = np.zeros(t.size)

    dphase = np.zeros(r.size)

    def data_gen():
        for i in range(t.size):
            print(f.E(t[i]+dt/2))
            yield i, tdse.calc.az(wf, atom, f, t[i])

            # ws.prop(wf, atom, f, t[i], dt/100)
            phase = np.angle(data)
            ws.prop(wf, atom, f, t[i], dt)
            #gps_ws.prop_comm(wf, uabs, f, t[i])
            dphase[:] = np.abs(np.angle(data) - phase)[0]
            #arr[:] = np.abs(arr)*np.exp(1.0j*(phase + dphase*100))

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
    #line7, = ax4.plot(r, orbs.grad_u(0), '.')
    line7, = ax4.plot(r, dphase)
    #ax4.set_xscale('log')
    ax4.set_ylim(1e-6, 100)
    ax4.set_yscale('log')

    line1,  = ax1.plot(t, az, label="az")
    line6, = ax1.plot(t[1:-1], np.abs(np.diff(z,2)/dt**2 - az[1:-1]), label="z")
    ax1.legend()

    line3, = ax2.plot(r, np.sum(np.abs(wf.asarray()), axis=0), label='Num')
    Z = 18
    ax2.plot(r, np.abs(-2*Z**1.5*r*np.exp(-r*Z)), label='Anal')
    ax2.legend()

    E = np.zeros(t.size)
    for i in range(t.size):
        E[i] = f.E(t[i])
    line4, = ax3.plot(t, E)
    line5, = ax3.plot([0,], [E[0],], '.')

    ax1.plot(t, np.abs(E))


    def run(data):
        it = data[0]
        az[it] = data[1]
        #prob[it] = wf.norm()#tdse.calc.ionization_prob(orbs)
        z[it] = wf.z()
        print("Z = ", z[it])
        print("Z = ", az[it])

        ax1.set_xlim(t[0], t[it])

        it = it+1

        #diff = np.abs(np.diff(z, 2)/dt**2 - az[1:-1])/E[1:-1]
        diff = np.diff(z, 2)/dt**2

        line1.set_ydata(az)
        line6.set_ydata(diff)

        #ax1.set_ylim(-np.max(diff[0:it]), np.max(diff[0:it]))
        ax1.set_ylim(-np.max(az[0:it]), np.max(az[0:it]))
        #ax1.set_yscale('log')
        #ax1.set_xscale('log')

        #data = orbs.grad_u(0)
        #line7.set_ydata(data)
        line7.set_ydata(dphase)
        #line7.set_ydata(com)
        #ax4.set_ylim(np.min(dphase[0]), np.max(dphase[0]))
        #ax4.set_ylim(np.min(com), np.max(com))
        #line6.set_ydata(z)

        #line3.set_ydata(np.sum(np.abs(wf.asarray()), axis=0))
        line3.set_ydata(np.abs(wf.asarray()[0]))
        #line5.set_data([t[it],],[E[it],])

        return (line3, line5, line6, line7),

    ani = animation.FuncAnimation(fig, run, data_gen, blit=False,
            interval=10, repeat=False)
    plt.legend()
    plt.show()
    return az

if __name__ == "__main__":
    calc_wf_az_t()
