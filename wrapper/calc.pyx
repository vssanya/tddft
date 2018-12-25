import numpy as np
cimport numpy as np

import numpy.fft

from wavefunc cimport ShWavefunc, ShNeWavefunc
from orbitals cimport ShOrbitals, ShNeOrbitals
from workspace cimport ShWavefuncWS, ShNeWavefuncWS, ShOrbitalsWS, ShNeOrbitalsWS
from field cimport Field
from atom cimport AtomCache, AtomNeCache

from calc_gpu cimport calc_wf_gpu_az
from wavefunc_gpu cimport ShWavefuncGPU


ctypedef fused WF:
    ShOrbitals
    ShNeOrbitals
    ShWavefunc
    ShWavefuncGPU
    ShNeWavefunc

ctypedef fused Orbs:
    ShOrbitals
    ShNeOrbitals

ctypedef fused WS:
    ShWavefuncWS
    ShNeWavefuncWS
    ShOrbitalsWS
    ShNeOrbitalsWS

ctypedef fused AC:
    AtomCache
    AtomNeCache

def az(WF wf, AC atom, Field field, double t):
    if WF is ShOrbitals:
        return calc_orbs_az(wf.cdata[0], atom.cdata[0], field.cdata, t)
    elif WF is ShNeOrbitals:
        return calc_orbs_az(wf.cdata[0], atom.cdata[0], field.cdata, t)
    elif WF is ShWavefuncGPU:
        return calc_wf_gpu_az(wf.cdata[0], atom.cdata[0], field.cdata, t)
    elif WF is ShWavefunc:
        return calc_wf_az(wf.cdata, atom.cdata[0], field.cdata, t)
    elif WF is ShNeWavefunc:
        return calc_wf_az(wf.cdata, atom.cdata[0], field.cdata, t)
    else:
        assert(False)

def az_with_polarization(ShWavefunc wf, AtomCache atom, double[:] Upol, double[:] dUpol_dr, Field field, double t):
    return calc_wf_az_with_polarization(wf.cdata, atom.cdata[0], &Upol[0], &dUpol_dr[0], field.cdata, t)

def az_ne(Orbs orbs, AC atom, Field field, double t, np.ndarray az = None):
    cdef double* res_ptr = NULL
    if orbs.is_root():
        if az is None:
            az = np.ndarray(orbs.atom.countOrbs, dtype=np.double)
        res_ptr = <double*>az.data


    calc_orbs_az_ne(orbs.cdata, atom.cdata[0], field.cdata, t, res_ptr)

    return az

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
