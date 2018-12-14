import numpy as np
import numpy.fft as fft

from . import const


UNIT = {
        'au': 1.0, # atomic unit

        's' : 1.0 / 2.418884326505e-17,
        'fs': 1.0 / 2.418884326505e-2,

        'm' : 1.0 / 5.2917721092e-11,
        'nm': 1.0 / 5.2917721092e-2,

        'eV': 1.602e-19 / 4.3597e-18,
        'W/cm2': 1.0 / 3.50944758e16,
        'MV/cm': 1.0 / 5.14220652e3
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

def r_osc(E, alpha, freq):
    return E/freq**2*(1 + alpha/4)

def r_max(E, alpha, freq):
    return 2*r_osc(E, alpha, freq)

def Up(E, alpha, freq):
    return (E/(2*freq))**2*(1 + alpha**2/4)

def Lmax(Ip, E, alpha, freq):
    pmax = 2*np.sqrt(Ip + 3.17*Up(E, alpha, freq))
    return pmax*r_max(E, alpha, freq)

"""
    I (W/cm^2)
    E (au)
"""
def I_to_E(I):
    return np.sqrt(I*UNIT['W/cm2'])

def E_to_I(E):
    return E**2 / UNIT['W/cm2']

def is_jupyter_notebook():
    try:
        cfg = get_ipython().config
        return True
    except Exception as e:
        return False
