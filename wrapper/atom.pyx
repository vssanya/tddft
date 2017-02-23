from wavefunc cimport SWavefunc
from grid cimport SGrid
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

    def get_init_orbs(self, SGrid grid):
        orbs = SOrbitals(self._data.ne, grid)
        self._data.init(orbs._data)
        return orbs

    def ort(self, SOrbitals orbs):
        self._data.ort(orbs._data)

def ground_state(SGrid grid):
    wf = SWavefunc(grid)
    atom_hydrogen_ground(wf.data)
    return wf
