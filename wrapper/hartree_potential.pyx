import numpy as np
cimport numpy as np

from wavefunc cimport SWavefunc
from orbitals cimport SOrbitals
from grid cimport SpGrid
from sphere_harmonics cimport YlmCache


def l0(SOrbitals orbs, np.ndarray[np.double_t, ndim=1] uh = None) -> np.ndarray:
    if uh is None:
        uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    hartree_potential_l0(orbs._data, &uh[0], &f[0])

    return uh

def wf_l0(SWavefunc wf, np.ndarray[np.double_t, ndim=1] uh = None) -> np.ndarray:
    if uh is None:
        uh = np.ndarray((wf.data.grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((wf.data.grid.n[0]), np.double)
    hartree_potential_wf_l0(wf.data, &uh[0], &f[0])

    return uh

def l1(SOrbitals orbs, np.ndarray[np.double_t, ndim=1] uh = None) -> np.ndarray:
    if uh is None:
        uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    hartree_potential_l1(orbs._data, &uh[0], &f[0])

    return uh

def l2(SOrbitals orbs, np.ndarray[np.double_t, ndim=1] uh = None) -> np.ndarray:
    if uh is None:
        uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    hartree_potential_l2(orbs._data, &uh[0], &f[0])

    return uh

def lda(int l, SOrbitals orbs, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=1] uxc = None, np.ndarray[np.double_t, ndim=1] n = None) -> np.ndarray:
    if uxc is None:
        uxc = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    else:
        assert(uxc.size == orbs._data.wf[0].grid.n[0])

    ux_lda(l, orbs._data, &uxc[0], grid.data, NULL, ylm_cache._data)

    return uxc

def lda_n(int l, np.ndarray[np.double_t, ndim=2] n, SpGrid grid, YlmCache ylm_cache, np.ndarray[np.double_t, ndim=1] uxc = None):
    if uxc is None:
        uxc = np.ndarray((grid.data.n[0]), np.double)
    else:
        assert(uxc.size == grid.data.n[0])

    ux_lda_n(l, grid.data, &n[0,0], &uxc[0], ylm_cache._data)

    return uxc

