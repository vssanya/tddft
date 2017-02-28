import numpy as np
cimport numpy as np

from wavefunc cimport SWavefunc
from orbitals cimport SOrbitals
from workspace cimport SKnWorkspace
from field cimport Field
from atom cimport Atom


def ionization_prob(SOrbitals orbs):
    return calc_ionization_prob(orbs._data)

def az(Atom atom, SWavefunc wf, Field field, double t):
    return calc_az(wf.data, field.data, atom._data.dudz, t)

def az_t(int Nt, SKnWorkspace ws, SWavefunc wf, Field field, double dt):
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] az = np.ndarray(Nt, dtype=np.double)
    calc_az_t(Nt, &az[0], ws.data, wf.data, field.data, dt)
    return az

def jrcd_t(Atom atom, SKnWorkspace ws, SWavefunc wf, Field field, int Nt, double dt, double t_smooth):
    return jrcd(ws.data, wf.data, field.data, atom._data.dudz, Nt, dt, t_smooth)

def smstep(double x, double x0, double x1):
    return smoothstep(x, x0, x1)
