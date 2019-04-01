import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import tdse.utils

def init_style_article():
    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)

    plt.style.use('article')

def figsize(scale, aspect=1, width_pt=469.0/2):
    """
    width_pt - get this from LaTeX using \\the\\textwidth
    """
    inches_per_pt = 1.0/72.27                # Convert pt to inch
    fig_width = width_pt*inches_per_pt*scale # width in inches
    fig_height = fig_width*aspect            # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def spectral_density(dt, freq, Sw, Nmax=100, scale=1, **kwargs):
    N = np.linspace(0, np.pi/dt, Sw.size) / freq
    plt.plot(N*scale, Sw, **kwargs)
    plt.yscale('log')
    plt.xlim(0, Nmax*scale)

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
