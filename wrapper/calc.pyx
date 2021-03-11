import numpy as np
cimport numpy as np

import pyfftw

from wavefunc cimport ShWavefunc, ShNeWavefunc
from orbitals cimport ShOrbitals, ShNeOrbitals
from workspace cimport ShWavefuncWS, ShNeWavefuncWS, ShOrbitalsWS, ShNeOrbitalsWS
from field cimport Field
from atom cimport ShAtomCache, ShNeAtomCache

# from calc_gpu cimport calc_wf_gpu_az
# from wavefunc_gpu cimport ShWavefuncGPU
from carray cimport DoubleArray2D


ctypedef fused WF:
    ShOrbitals
    ShNeOrbitals
    ShWavefunc
    #ShWavefuncGPU
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
    #elif WF is ShWavefuncGPU and AC is ShAtomCache:
        #return calc_wf_gpu_az(wf.cdata[0], atom.cdata[0], field.cdata, t)
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
        return calc_orbs_az_ne_Vee_1(orbs.cdata, array_uee.cdata[0], array_dueedr.cdata[0], atom.cdata[0], field.cdata, t, res_ptr)
    elif Orbs is ShNeOrbitals and AC is ShNeAtomCache:
        return calc_orbs_az_ne_Vee_1(orbs.cdata, array_uee.cdata[0], array_dueedr.cdata[0], atom.cdata[0], field.cdata, t, res_ptr)
    else:
        assert(False)

def az_ne_Vee_2(Orbs orbs, AC atom, Field field, double t, np.ndarray Uee, np.ndarray dUeedr, np.ndarray az = None):
    cdef DoubleArray2D array_uee    = DoubleArray2D(Uee)
    cdef DoubleArray2D array_dueedr = DoubleArray2D(dUeedr)

    cdef double* res_ptr = NULL
    if orbs.is_root():
        if az is None:
            az = np.ndarray(orbs.atom.countOrbs, dtype=np.double)
        res_ptr = <double*>az.data

    if Orbs is ShOrbitals and AC is ShAtomCache:
        return calc_orbs_az_ne_Vee_2(orbs.cdata, array_uee.cdata[0], array_dueedr.cdata[0], atom.cdata[0], field.cdata, t, res_ptr)
    elif Orbs is ShNeOrbitals and AC is ShNeAtomCache:
        return calc_orbs_az_ne_Vee_2(orbs.cdata, array_uee.cdata[0], array_dueedr.cdata[0], atom.cdata[0], field.cdata, t, res_ptr)
    else:
        assert(False)

def az_ne_Vee(Orbs orbs, AC atom, Field field, double t, np.ndarray Uee, np.ndarray dUeedr, np.ndarray az = None, int l = 0):
    if l == 0:
        return az_ne_Vee_0(orbs, atom, field, t, Uee, dUeedr, az)
    elif l == 1:
        return az_ne_Vee_1(orbs, atom, field, t, Uee, dUeedr, az)
    elif l == 2:
        return az_ne_Vee_2(orbs, atom, field, t, Uee, dUeedr, az)
    else:
        assert(False)

def smstep(double x, double x0, double x1):
    return smoothstep(x, x0, x1)

@np.vectorize
def ionization_probability_ppt(int l, int m, double Cnl, double Ip, int Z, double E, double freq):
    return w_ppt(l, m, Cnl, Ip, Z, E, freq)

@np.vectorize
def ionization_probability_adk(int l, int m, double Cnl, double Ip, int Z, double E, double freq):
    return w_adk(l, m, Cnl, Ip, Z, E, freq)

@np.vectorize
def ionization_probability_ppt_Qc(int l, int m, double Cnl, double Ip, int Z, double E, double freq):
    return w_ppt_Qc(l, m, Cnl, Ip, Z, E, freq)

@np.vectorize
def ionization_probability_tl_exp(double Ip, int Z, double E, double alpha):
    return w_tl_exp(Ip, Z, E, alpha)

def int_ppt(double x, double m):
    return int_func_res(x, m)

def mask_calc(N, width):
    cdef i

    res = np.ndarray(N, dtype=np.double)
    for i in range(N):
        res[i] = 1 - smstep(i, N*(1.0-width), N-1)

    return res

def spectral_density_phase(az, double dt, np.ndarray[double, ndim=1] mask = None, mask_width=0.0) -> np.ndarray:
    if mask is None:
        mask = mask_calc(az.shape[-1], mask_width)

    return pyfftw.interfaces.numpy_fft.rfft(az*mask)

def spectral_density(az, double dt, np.ndarray[double, ndim=1] mask = None, mask_width=0.0) -> np.ndarray:
    if mask is None:
        mask = mask_calc(az.shape[-1], mask_width)

    return np.abs(pyfftw.interfaces.numpy_fft.rfft(az*mask)*dt)**2

def jrcd(np.ndarray[double, ndim=1] az, double dt, np.ndarray[double, ndim=1] mask = None, mask_width=0.0) -> np.float64:
    if mask is None:
        mask = mask_calc(az.size, mask_width)

    return np.sum(az*mask)*dt

def setGpuDevice(int id):
    return selectGpuDevice(id)

def w_from_aw(double dt, Sw):
    return np.linspace(0, np.pi/dt, Sw.size)

def N_from_aw(double dt, double freq, Sw):
    return w_from_aw(dt, Sw) / freq

def search_hhg_argmin(double dt, double freq, Sw, interval=(0,1)):
    dN = np.pi / dt / Sw.size / freq
    start = int(interval[0] / dN)
    end = int(interval[1] / dN)
    if start == end:
        return start
    else:
        return start + np.argmin(Sw[start:end])

def j_interval(double dt, double freq, az, mask_width=0.1, intervals=((0,1),(1,2))):
    cdef int i = 0

    mask = mask_calc(az.shape[-1], mask_width)

    aw = pyfftw.interfaces.numpy_fft.rfft(az*mask)
    
    start = search_hhg_argmin(dt, freq, np.abs(aw), intervals[0])
    end = search_hhg_argmin(dt, freq, np.abs(aw), intervals[1])

    print(start)
    print(end)
    
    aw[:start] = 0.0
    aw[end:] = 0.0

    return pyfftw.interfaces.numpy_fft.irfft(aw) 

def calc_field_return_rmax(Field field, double dt, double r_atom = 1.0):
    cdef np.ndarray[double, ndim=1] t = field.get_t(dt)
    cdef np.ndarray[double, ndim=1] E = field.E(t)
    return calc_r_max(E.size, &E[0], dt, r_atom)

def calc_field_rmax(Field field, double dt):
    cdef np.ndarray[double, ndim=1] t = field.get_t(dt)
    cdef np.ndarray[double, ndim=1] E = field.E(t)
    return calc_r_max_without_return(E.size, &E[0], dt)

def calc_field_prmax(Field field, double dt, double r_max):
    cdef np.ndarray[double, ndim=1] t = field.get_t(dt)
    cdef np.ndarray[double, ndim=1] E = field.E(t)
    return calc_pr_max(E.size, &E[0], dt, r_max)
