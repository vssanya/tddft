import numpy as np
cimport numpy as np

from wavefunc cimport SWavefunc
from orbitals cimport SOrbitals
from workspace cimport SKnWorkspace, SOrbsWorkspace
from field cimport Field
from atom cimport Atom


ctypedef fused WF:
    SOrbitals
    SWavefunc

ctypedef fused WS:
    SKnWorkspace
    SOrbsWorkspace

def ionization_prob(WF wf):
    if WF is SOrbitals:
        return calc_orbs_ionization_prob(wf._data)
    else:
        return calc_wf_ionization_prob(wf.data)

def az(WF wf, Atom atom, Field field, double t):
    if WF is SOrbitals:
        return calc_orbs_az(wf._data, atom._data, field.data, t)
    else:
        return calc_wf_az(wf.data, atom._data, field.data, t)

def az_ne(SOrbitals orbs, Field field, double t, np.ndarray[double, ndim=1, mode='c'] az = None):
    cdef double* res_ptr = NULL
    if orbs.is_root():
        if az is None:
            az = np.ndarray(orbs._data.atom.n_orbs, dtype=np.double)
        res_ptr = <double*>az.data

    calc_orbs_az_ne(orbs._data, field.data, t, res_ptr)

    return az

def jrcd(Atom atom, WS ws, WF wf, Field field, int Nt, double dt, double t_smooth):
    if WF is SOrbitals and WS is SOrbsWorkspace:
        return calc_orbs_jrcd(ws._data, wf._data, atom._data, field.data, Nt, dt, t_smooth)
    elif WF is SWavefunc and WS is SKnWorkspace:
        return calc_wf_jrcd(ws.data, wf.data, atom._data, field.data, Nt, dt, t_smooth)
    else:
        assert(False)

def smstep(double x, double x0, double x1):
    return smoothstep(x, x0, x1)
