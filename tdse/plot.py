import numpy as np
import matplotlib.pyplot as plt

import tdse.utils


def spectral_density(dt, freq, Sw, Nmax=100, **kwargs):
    N = np.linspace(0, np.pi/dt, Sw.size) / freq
    plt.plot(N, Sw, **kwargs)
    plt.yscale('log')
    plt.xlim(0, Nmax)

    plt.xlabel('Harmonic order')
    plt.ylabel('HHG spectrum (a.u.)')


def spectral_density_ev(dt, freq, Sw, Nmax=100, **kwargs):
    E = tdse.utils.unit_to(np.linspace(0, np.pi/dt, Sw.size), 'au', 'eV')
    Emax = tdse.utils.unit_to(Nmax*freq, 'au', 'eV')
    plt.plot(E, Sw, **kwargs)
    plt.yscale('log')
    plt.xlim(0, Emax)

    plt.xlabel('Photon energy (eV)')
    plt.ylabel('HHG spectrum (a.u.)')
