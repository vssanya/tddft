import numpy as np
cimport numpy as np

from wavefunc cimport SWavefunc
from grid cimport ShGrid
from orbitals cimport SOrbitals


cdef class Atom:
    def __cinit__(self, id):
        if id == 'H':
            self._data = &atom_hydrogen
        elif id == 'Ne':
            self._data = &atom_neon
        elif id == 'Ar':
            self._data = &atom_argon
        else:
            assert(False, 'Atom {} is not exist'.format(id))

    def get_init_orbs(self, ShGrid grid):
        orbs = SOrbitals(self._data.ne, grid)
        self._data.init(orbs._data)
        return orbs

    def get_ground_state(self, ShGrid grid, filename):
        orbs = SOrbitals(self._data.ne, grid)

        arr = orbs.asarray()
        arr[:] = 0.0

        ground_state = np.load(filename)
        for l in range(ground_state.shape[1]):
            arr[:,l,:] = ground_state[:,l,:]

        return orbs

    def ort(self, SOrbitals orbs):
        self._data.ort(orbs._data)

def ground_state(ShGrid grid) -> SWavefunc:
    wf = SWavefunc(grid)
    atom_hydrogen_ground(wf.data)
    return wf
