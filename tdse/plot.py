import numpy as np
import matplotlib.pyplot as plt


def spectral_density(dt, freq, Sw, Nmax=100, **kwargs):
    N = np.linspace(0, np.pi/dt, Sw.size) / freq
    plt.plot(N, Sw, **kwargs)
    plt.yscale('log')
    plt.xlim(0, Nmax)

    plt.xlabel('Harmonic order')
    plt.ylabel('HHG spectrum (a.u.)')
