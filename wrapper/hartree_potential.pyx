import numpy as np
cimport numpy as np

from wavefunc cimport ShWavefunc
from orbitals cimport Orbitals
from grid cimport SpGrid
from sphere_harmonics cimport YlmCache

cdef class Uxc:
    @staticmethod
    cdef Uxc from_c_func(potential_xc_f func):
        obj = <Uxc>Uxc.__new__(Uxc)
        obj.cdata = func
        return obj

    def calc_l(self, int l, Orbitals orbs, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=1] uxc = None, np.ndarray[np.double_t, ndim=2] n = None) -> np.ndarray:
        if uxc is None:
            uxc = np.ndarray((orbs.cdata.wf[0].grid.n[0]), np.double)
        else:
            assert(uxc.size == orbs.cdata.wf[0].grid.n[0])

        uxc_calc_l(self.cdata, l, orbs.cdata, &uxc[0], grid.data, &n[0,0], NULL, ylm_cache.cdata)

        return uxc

    def calc_l0(self, int l, Orbitals orbs, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=1] uxc = None, np.ndarray[np.double_t, ndim=2] n = None) -> np.ndarray:
        if uxc is None:
            uxc = np.ndarray((orbs.cdata.wf[0].grid.n[0]), np.double)
        else:
            assert(uxc.size == orbs.cdata.wf[0].grid.n[0])

        uxc_calc_l0(self.cdata, l, orbs.cdata, &uxc[0], grid.data, &n[0,0], NULL, ylm_cache.cdata)

        return uxc

UXC_LB    = Uxc.from_c_func(uxc_lb)
UXC_LDA   = Uxc.from_c_func(uxc_lda)
UXC_LDA_X = Uxc.from_c_func(uxc_lda_x)

@np.vectorize
def uc_lda(double n):
    return uc_lda_func(n)


def potential(Orbitals orbs, int l = 0, np.ndarray[np.double_t, ndim=1] uh = None, int order=3) -> np.ndarray:
    if uh is None:
        uh = np.ndarray((orbs.cdata.wf[0].grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((orbs.cdata.wf[0].grid.n[0]), np.double)
    hartree_potential(orbs.cdata, l, &uh[0], &uh[0], &f[0], order)

    return uh


def wf_l0(ShWavefunc wf, np.ndarray[np.double_t, ndim=1] uh = None, int order=3) -> np.ndarray:
    if uh is None:
        uh = np.ndarray((wf.cdata.grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((wf.cdata.grid.n[0]), np.double)
    hartree_potential_wf_l0(wf.cdata, &uh[0], &f[0], order)

    return uh
