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

def spectral_density(dt, freq, Sw, Nmax=100, scale=1, convert_x=None, **kwargs):
    w = np.linspace(0, np.pi/dt, Sw.size)
    if convert_x is not None:
        w = convert_x(w)

    plt.plot(w*scale, Sw, **kwargs)
    plt.yscale('log')
    plt.xlim(0, Nmax*scale)

    plt.xlabel('Frequency (a.u.)')
    plt.ylabel('HHG spectrum (a.u.)')

    return w


def spectral_density_n(dt, freq, Sw, Nmax=100, scale=1, **kwargs):
    N = spectral_density(dt, freq, Sw, Nmax, scale, convert_x=lambda x: x/freq, **kwargs)
    plt.xlabel('Harmonic order')

    return N

def spectral_density_ev(dt, freq, Sw, Nmax=100, unit='eV', **kwargs):
    Emax = tdse.utils.unit_to(Nmax*freq, 'au', unit)
    E = spectral_density(dt, freq, Sw, Emax, convert_x=lambda x: tdse.utils.unit_to(x, 'au', unit), **kwargs)
    plt.xlabel('Photon energy (eV)')

    return E

def I_label(value, power):
    return r"$I = {0} \times 10^{{ {1} }}$ W/cm$^2$".format(value, power)
