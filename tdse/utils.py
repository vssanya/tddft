import numpy as np
from . import const

UNIT = {
        'au': 1.0, # atomic unit

        's' : 1 / 2.418884326505e-17,
        'fs': 1 / 2.418884326505e-2,

        'm': 1 / 5.2917721092e-11,
        'nm': 1 / 5.2917721092e-2,

        'eV': 1.602e-19 / 4.3597e-18
}

@np.vectorize
def unit_to(value, u_from='au', u_to='au'):
    return UNIT[u_from]*value/UNIT[u_to]

def t_fwhm(fwhm, u='fs', u_to='au'):
    return unit_to(fwhm, u, u_to)/np.sqrt(2*np.log(2))

def length_to_freq(length, u='nm', u_to='au'):
    freq_au = 2*np.pi*const.C / unit_to(length, u)
    return unit_to(freq_au, u_to=u_to)

def t_shift(tp, I0, Imin):
    return np.sqrt(0.5*tp**2*np.log(I0/Imin))

def r_max(E, alpha, freq):
    return 3*E/freq**2*(1 + alpha/4)

"""
    I (W/cm^2)
    E (au)
"""
def I_to_E(I):
    a = 1e14 / (5.34e-2)**2
    return np.sqrt(I/a)

def E_to_I(E):
    a = 1e14 / (5.34e-2)**2
    return a * E**2
