import numpy as np
cimport numpy as np

import numpy.fft

from wavefunc cimport ShWavefunc, ShNeWavefunc
from orbitals cimport Orbitals
from workspace cimport SKnWorkspace, SOrbsWorkspace
from field cimport Field
from atom cimport AtomCache, AtomNeCache

from calc_gpu cimport calc_wf_gpu_az
from wavefunc_gpu cimport ShWavefuncGPU


ctypedef fused WF:
    Orbitals
    ShWavefunc
    ShWavefuncGPU
    ShNeWavefunc

ctypedef fused WS:
    SKnWorkspace
    SOrbsWorkspace

ctypedef fused AC:
    AtomCache
    AtomNeCache

def ionization_prob(WF wf):
    if WF is Orbitals:
        return calc_orbs_ionization_prob(wf.cdata)
    elif WF is ShWavefunc:
        return calc_wf_ionization_prob(wf.cdata)
    else:
        assert(False)

def az(WF wf, AC atom, Field field, double t):
    if WF is Orbitals:
        return calc_orbs_az(wf.cdata[0], atom.cdata[0], field.cdata, t)
    elif WF is ShWavefuncGPU:
        return calc_wf_gpu_az(wf.cdata[0], atom.cdata[0], field.cdata, t)
    elif WF is ShWavefunc or WF is ShNeWavefunc:
        return calc_wf_az(wf.cdata, atom.cdata[0], field.cdata, t)
    else:
        assert(False)

def az_with_polarization(ShWavefunc wf, AtomCache atom, double[:] Upol, double[:] dUpol_dr, Field field, double t):
    return calc_wf_az_with_polarization(wf.cdata, atom.cdata[0], &Upol[0], &dUpol_dr[0], field.cdata, t)

def az_ne(Orbitals orbs, AtomCache atom, Field field, double t, np.ndarray[double, ndim=1, mode='c'] az = None):
    cdef double* res_ptr = NULL
    if orbs.is_root():
        if az is None:
            az = np.ndarray(orbs.atom.countOrbs, dtype=np.double)
        res_ptr = <double*>az.data

    calc_orbs_az_ne(orbs.cdata, atom.cdata[0], field.cdata, t, res_ptr)

    return az

def jrcd(AtomCache atom, WS ws, WF wf, Field field, int Nt, double dt, double t_smooth):
    if WF is Orbitals and WS is SOrbsWorkspace:
        return calc_orbs_jrcd(ws.cdata[0], wf.cdata[0], atom.cdata[0], field.cdata, Nt, dt, t_smooth)
    elif WF is ShWavefunc and WS is SKnWorkspace:
        return calc_wf_jrcd(ws.cdata, wf.cdata, atom.cdata[0], field.cdata, Nt, dt, t_smooth)
    else:
        assert(False)

def smstep(double x, double x0, double x1):
    return smoothstep(x, x0, x1)

def spectral_density(np.ndarray[double, ndim=1] az, double dt, np.ndarray[double, ndim=1] mask = None, mask_width=0.0) -> np.ndarray:
    cdef i

    if mask is None:
        mask = np.ndarray(az.size, dtype=np.double)
        for i in range(mask.size):
            mask[i] = 1 - smstep(i, mask.size*(1.0-mask_width), mask.size-1)

    return np.abs(numpy.fft.rfft(az*mask)*dt)**2

def setGpuDevice(int id):
    return selectGpuDevice(id)
