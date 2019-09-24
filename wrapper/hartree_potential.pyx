import numpy as np
cimport numpy as np

from wavefunc cimport ShWavefunc, ShNeWavefunc
from orbitals cimport ShOrbitals, ShNeOrbitals
from grid cimport SpGrid, cShGrid, cShNeGrid
from sphere_harmonics cimport YlmCache

ctypedef fused Orbs:
    ShOrbitals
    ShNeOrbitals

ctypedef fused Wf:
    ShWavefunc
    ShNeWavefunc

cdef class Uxc:
    @staticmethod
    cdef Uxc from_c_func(potential_xc_f func, str name):
        obj = <Uxc>Uxc.__new__(Uxc)
        obj.cdata = func
        obj.name = name

        return obj

    def calc_l(self, int l, Orbs orbs, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=1] uxc = None, np.ndarray[np.double_t, ndim=2] n = None) -> np.ndarray:
        if uxc is None:
            uxc = np.ndarray((orbs.cdata.wf[0].grid.n[0]), np.double)
        else:
            assert(uxc.size == orbs.cdata.wf[0].grid.n[0])

        if Orbs is ShOrbitals:
            XCPotential[cShGrid].calc_l(self.cdata, l, orbs.cdata, &uxc[0], grid.data, &n[0,0], NULL, ylm_cache.cdata)
        elif Orbs is ShNeOrbitals:
            XCPotential[cShNeGrid].calc_l(self.cdata, l, orbs.cdata, &uxc[0], grid.data, &n[0,0], NULL, ylm_cache.cdata)

        return uxc

    def calc_l0(self, int l, Orbs orbs, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=1] uxc = None, np.ndarray[np.double_t, ndim=2] n = None) -> np.ndarray:
        if uxc is None:
            uxc = np.ndarray((orbs.cdata.wf[0].grid.n[0]), np.double)
        else:
            assert(uxc.size == orbs.cdata.wf[0].grid.n[0])

        if Orbs is ShOrbitals:
            XCPotential[cShGrid].calc_l0(self.cdata, l, orbs.cdata, &uxc[0], grid.data, &n[0,0], NULL, ylm_cache.cdata)
        elif Orbs is ShNeOrbitals:
            XCPotential[cShNeGrid].calc_l0(self.cdata, l, orbs.cdata, &uxc[0], grid.data, &n[0,0], NULL, ylm_cache.cdata)

        return uxc

    def write_params(self, params_grp):
        params_grp.attrs['Uxc_type'] = self.name

UXC_LB    = Uxc.from_c_func(uxc_lb, "LB")
UXC_LDA   = Uxc.from_c_func(uxc_lda, "LDA")
UXC_LDA_X = Uxc.from_c_func(uxc_lda_x, "LDA_X")

@np.vectorize
def uc_lda(double n):
    return uc_lda_func(n)


def potential(Orbs orbs, int l = 0, np.ndarray uh = None, np.ndarray uh_local = None, int order=3) -> np.ndarray:
    if uh is None:
        uh = np.ndarray((orbs.cdata.grid.n[0]), np.double)

    if uh_local is None:
        uh_local = np.ndarray((orbs.cdata.grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((orbs.cdata.grid.n[0]), np.double)
    if Orbs is ShOrbitals:
        HartreePotential[cShGrid].calc(orbs.cdata, l, <double*>uh.data, <double*>uh_local.data, &f[0], order)
    elif Orbs is ShNeOrbitals:
        HartreePotential[cShNeGrid].calc(orbs.cdata, l, <double*>uh.data, <double*>uh_local.data, &f[0], order)

    return uh

def potential_int_func(Orbs orbs, int l = 0) -> np.ndarray:
    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((orbs.cdata.wf[0].grid.n[0]), np.double)
    if Orbs is ShOrbitals:
        HartreePotential[cShGrid].calc_int_func(orbs.cdata, l, &f[0])
    elif Orbs is ShNeOrbitals:
        HartreePotential[cShNeGrid].calc_int_func(orbs.cdata, l, &f[0])

    return f


def wf_l0(Wf wf, np.ndarray uh = None, int order=3) -> np.ndarray:
    if uh is None:
        uh = np.ndarray((wf.cdata.grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((wf.cdata.grid.n[0]), np.double)
    if Wf is ShWavefunc:
        HartreePotential[cShGrid].calc_wf_l0(wf.cdata, <double*>uh.data, &f[0], order)
    elif Wf is ShNeWavefunc:
        HartreePotential[cShNeGrid].calc_wf_l0(wf.cdata, <double*>uh.data, &f[0], order)

    return uh
