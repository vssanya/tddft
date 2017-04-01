import numpy as np
cimport numpy as np

import mpi4py

from types cimport cdouble
from wavefunc cimport SWavefunc
from grid cimport ShGrid


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

def ground_state(ShGrid grid) -> SWavefunc:
    wf = SWavefunc(grid)
    atom_hydrogen_ground(wf.data)
    return wf
