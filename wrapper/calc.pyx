import numpy as np
cimport numpy as np

import numpy.fft

from wavefunc cimport ShWavefunc, ShNeWavefunc
from orbitals cimport ShOrbitals, ShNeOrbitals
from workspace cimport ShWavefuncWS, ShNeWavefuncWS, ShOrbitalsWS, ShNeOrbitalsWS
from field cimport Field
from atom cimport ShAtomCache, ShNeAtomCache

from calc_gpu cimport calc_wf_gpu_az
from wavefunc_gpu cimport ShWavefuncGPU
from carray cimport DoubleArray2D


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
    ShAtomCache
    ShNeAtomCache

def az(WF wf, AC atom, Field field, double t):
    if WF is ShOrbitals and AC is ShAtomCache:
        return calc_orbs_az(wf.cdata[0], atom.cdata[0], field.cdata, t)
    elif WF is ShNeOrbitals and AC is ShNeAtomCache:
        return calc_orbs_az(wf.cdata[0], atom.cdata[0], field.cdata, t)
    elif WF is ShWavefuncGPU and AC is ShAtomCache:
        return calc_wf_gpu_az(wf.cdata[0], atom.cdata[0], field.cdata, t)
    elif WF is ShWavefunc and AC is ShAtomCache:
        return calc_wf_az(wf.cdata, atom.cdata[0], field.cdata, t)
    elif WF is ShNeWavefunc and AC is ShNeAtomCache:
        return calc_wf_az(wf.cdata, atom.cdata[0], field.cdata, t)
    else:
        assert(False)

def wf_az_p(WF wf_p, WF wf_g, AC atom, int lmax = -1):
    if WF is ShWavefunc and AC is ShAtomCache:
        return calc_wf_az(wf_p.cdata, wf_g.cdata, atom.cdata[0], lmax)
    elif WF is ShNeWavefunc and AC is ShNeAtomCache:
        return calc_wf_az(wf_p.cdata, wf_g.cdata, atom.cdata[0], lmax)
    else:
        assert(False)

def az_with_polarization(ShWavefunc wf, ShAtomCache atom, double[:] Upol, double[:] dUpol_dr, Field field, double t):
    return calc_wf_az_with_polarization(wf.cdata, atom.cdata[0], &Upol[0], &dUpol_dr[0], field.cdata, t)

def az_ne(Orbs orbs, AC atom, Field field, double t, np.ndarray az = None):
    cdef double* res_ptr = NULL
    if orbs.is_root():
        if az is None:
            az = np.ndarray(orbs.atom.countOrbs, dtype=np.double)
        res_ptr = <double*>az.data


    if Orbs is ShOrbitals and AC is ShAtomCache:
        calc_orbs_az_ne(orbs.cdata, atom.cdata[0], field.cdata, t, res_ptr)
    elif Orbs is ShNeOrbitals and AC is ShNeAtomCache:
        calc_orbs_az_ne(orbs.cdata, atom.cdata[0], field.cdata, t, res_ptr)
    else:
        assert(False)

    return az

def az_ne_Vee_0(Orbs orbs, AC atom, Field field, double t, np.ndarray Uee, np.ndarray dUeedr, np.ndarray az = None):
    cdef DoubleArray2D array_uee    = DoubleArray2D(Uee)
    cdef DoubleArray2D array_dueedr = DoubleArray2D(dUeedr)

    cdef double* res_ptr = NULL
    if orbs.is_root():
        if az is None:
            az = np.ndarray(orbs.atom.countOrbs, dtype=np.double)
        res_ptr = <double*>az.data

    if Orbs is ShOrbitals and AC is ShAtomCache:
        return calc_orbs_az_ne_Vee_0(orbs.cdata, array_uee.cdata[0], array_dueedr.cdata[0], atom.cdata[0], field.cdata, t, res_ptr)
    elif Orbs is ShNeOrbitals and AC is ShNeAtomCache:
        return calc_orbs_az_ne_Vee_0(orbs.cdata, array_uee.cdata[0], array_dueedr.cdata[0], atom.cdata[0], field.cdata, t, res_ptr)
    else:
        assert(False)

def az_ne_Vee_1(Orbs orbs, AC atom, Field field, double t, np.ndarray Uee, np.ndarray dUeedr, np.ndarray az = None):
    cdef DoubleArray2D array_uee    = DoubleArray2D(Uee)
    cdef DoubleArray2D array_dueedr = DoubleArray2D(dUeedr)

    cdef double* res_ptr = NULL
    if orbs.is_root():
        if az is None:
            az = np.ndarray(orbs.atom.countOrbs, dtype=np.double)
        res_ptr = <double*>az.data

    if Orbs is ShOrbitals and AC is ShAtomCache:
        return calc_orbs_az_ne_Vee_0(orbs.cdata, array_uee.cdata[0], array_dueedr.cdata[0], atom.cdata[0], field.cdata, t, res_ptr)
    elif Orbs is ShNeOrbitals and AC is ShNeAtomCache:
        return calc_orbs_az_ne_Vee_0(orbs.cdata, array_uee.cdata[0], array_dueedr.cdata[0], atom.cdata[0], field.cdata, t, res_ptr)
    else:
        assert(False)

def smstep(double x, double x0, double x1):
    return smoothstep(x, x0, x1)

@np.vectorize
def ionization_probability_ppt(int l, int m, double Cnl, double Ip, int Z, double E, double freq):
    return w_ppt(l, m, Cnl, Ip, Z, E, freq)

def int_ppt(double x, double m):
    return int_func_res(x, m)

def spectral_density(np.ndarray[double, ndim=1] az, double dt, np.ndarray[double, ndim=1] mask = None, mask_width=0.0) -> np.ndarray:
    cdef i

    if mask is None:
        mask = np.ndarray(az.size, dtype=np.double)
        for i in range(mask.size):
            mask[i] = 1 - smstep(i, mask.size*(1.0-mask_width), mask.size-1)

    return np.abs(numpy.fft.rfft(az*mask)*dt)**2

def setGpuDevice(int id):
    return selectGpuDevice(id)
