import numpy as np
cimport numpy as np

from orbitals cimport SOrbitals


def l0(SOrbitals orbs):
    cdef np.ndarray[np.double_t, ndim=1] uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    hartree_potential_l0(orbs._data, &uh[0])
    return uh

def l1(SOrbitals orbs):
    cdef np.ndarray[np.double_t, ndim=1] uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    hartree_potential_l1(orbs._data, &uh[0])
    return uh

def l2(SOrbitals orbs):
    cdef np.ndarray[np.double_t, ndim=1] uh = np.ndarray((orbs._data.wf[0].grid.n[0]), np.double)
    hartree_potential_l2(orbs._data, &uh[0])
    return uh
