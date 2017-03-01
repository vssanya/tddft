import numpy as np
cimport numpy as np

from orbitals cimport SOrbitals
from grid cimport SpGrid


def l0(SOrbitals orbs, np.ndarray[np.double_t, ndim=1] uh = None):
    if uh is None:
        uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)

    hartree_potential_l0(orbs._data, &uh[0])

    return uh

def l1(SOrbitals orbs, np.ndarray[np.double_t, ndim=1] uh = None):
    if uh is None:
        uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)

    hartree_potential_l1(orbs._data, &uh[0])

    return uh

def l2(SOrbitals orbs, np.ndarray[np.double_t, ndim=1] uh = None):
    if uh is None:
        uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)

    hartree_potential_l2(orbs._data, &uh[0])

    return uh

def lda(int l, SOrbitals orbs, SpGrid grid, np.ndarray[np.double_t, ndim=1] uxc = None):
    if uxc is None:
        uxc = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    else:
        assert(uxc.size == orbs._data.wf[0].grid.n[0])

    ux_lda(l, orbs._data, &uxc[0], grid.data)

    return uxc

