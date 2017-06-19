import numpy as np
cimport numpy as np

import mpi4py

from types cimport cdouble
from wavefunc cimport SWavefunc
from grid cimport ShGrid


cdef class Atom:
    def __cinit__(self, id):
        if id == 'H':
            self.cdata = &atom_hydrogen
        elif id == 'H_smooth':
            self.cdata = &atom_hydrogen_smooth
        elif id == 'Ne':
            self.cdata = &atom_neon
        elif id == 'Ar':
            self.cdata = &atom_argon
        elif id == 'Ar_gs':
            self.cdata = &atom_argon_gs
        else:
            assert(False, 'Atom {} is not exist'.format(id))

def ground_state(ShGrid grid) -> SWavefunc:
    wf = SWavefunc(grid)
    atom_hydrogen_ground(wf.cdata)
    return wf
