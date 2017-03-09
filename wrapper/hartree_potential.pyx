import numpy as np
cimport numpy as np

from wavefunc cimport SWavefunc
from orbitals cimport SOrbitals
from grid cimport SpGrid


def l0(SOrbitals orbs, np.ndarray[np.double_t, ndim=1] uh = None):
    if uh is None:
        uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    hartree_potential_l0(orbs._data, &uh[0], &f[0])

    return uh

def wf_l0(SWavefunc wf, np.ndarray[np.double_t, ndim=1] uh = None):
    if uh is None:
        uh = np.ndarray((wf.data.grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((wf.data.grid.n[0]), np.double)
    hartree_potential_wf_l0(wf.data, &uh[0], &f[0])

    return uh

def l1(SOrbitals orbs, np.ndarray[np.double_t, ndim=1] uh = None):
    if uh is None:
        uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    hartree_potential_l1(orbs._data, &uh[0], &f[0])

    return uh

def l2(SOrbitals orbs, np.ndarray[np.double_t, ndim=1] uh = None):
    if uh is None:
        uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)

    cdef np.ndarray[np.double_t, ndim=1] f = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    hartree_potential_l2(orbs._data, &uh[0], &f[0])

    return uh

def lda(int l, SOrbitals orbs, SpGrid grid, np.ndarray[np.double_t, ndim=1] uxc = None):
    if uxc is None:
        uxc = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    else:
        assert(uxc.size == orbs._data.wf[0].grid.n[0])

    ux_lda(l, orbs._data, &uxc[0], grid.data)

    return uxc

def lda_n(int l, np.ndarray[np.double_t, ndim=2] n, SpGrid grid, np.ndarray[np.double_t, ndim=1] uxc = None):
    if uxc is None:
        uxc = np.ndarray((grid.data.n[0]), np.double)
    else:
        assert(uxc.size == grid.data.n[0])

    ux_lda_n(l, grid.data, &n[0,0], &uxc[0])

    return uxc

