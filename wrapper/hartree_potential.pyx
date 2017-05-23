import numpy as np
cimport numpy as np

from wavefunc cimport SWavefunc
from orbitals cimport SOrbitals
from grid cimport SpGrid
from sphere_harmonics cimport YlmCache


@np.vectorize
def uc_lda(double n):
    return uc_lda_func(n)

def potential(SOrbitals orbs, int l = 0, np.ndarray[np.double_t, ndim=1] uh = None, int order=3) -> np.ndarray:
    if uh is None:
        uh = np.ndarray((orbs.cdata.wf[0].grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((orbs.cdata.wf[0].grid.n[0]), np.double)
    hartree_potential(orbs.cdata, l, &uh[0], &uh[0], &f[0], order)

    return uh

def wf_l0(SWavefunc wf, np.ndarray[np.double_t, ndim=1] uh = None, int order=3) -> np.ndarray:
    if uh is None:
        uh = np.ndarray((wf.cdata.grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((wf.cdata.grid.n[0]), np.double)
    hartree_potential_wf_l0(wf.cdata, &uh[0], &f[0], order)

    return uh

def lda(int l, SOrbitals orbs, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=1] uxc = None, np.ndarray[np.double_t, ndim=1] n = None) -> np.ndarray:
    if uxc is None:
        uxc = np.ndarray((orbs.cdata.wf[0].grid.n[0]), np.double)
    else:
        assert(uxc.size == orbs.cdata.wf[0].grid.n[0])

    uxc_lb(l, orbs.cdata, &uxc[0], grid.data, NULL, NULL, ylm_cache.cdata)

    return uxc
