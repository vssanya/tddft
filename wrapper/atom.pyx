import numpy as np
cimport numpy as np

import mpi4py

from types cimport cdouble
from wavefunc cimport SWavefunc
from grid cimport ShGrid
from orbitals cimport SOrbitals, orbitals_set_init_state


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

    def get_ground_state(self, ShGrid grid, filename, comm = None):
        """
        [MPI support]
        """

        orbs = SOrbitals(self._data.n_orbs, grid, comm)

        arr = orbs.asarray()
        arr[:] = 0.0

        cdef np.ndarray[np.complex_t, ndim=3] data
        cdef cdouble* data_ptr = NULL

        if not orbs.is_mpi() or orbs._data.mpi_rank == 0:
            data = np.load(filename)
            l_max = data.shape[1]
            data_ptr = <cdouble*>data.data
        else:
            l_max = None

        if orbs.is_mpi():
            l_max = comm.bcast(l_max)

        orbitals_set_init_state(orbs._data, data_ptr, l_max)

        return orbs

    def ort(self, SOrbitals orbs):
        self._data.ort(orbs._data)

def ground_state(ShGrid grid) -> SWavefunc:
    wf = SWavefunc(grid)
    atom_hydrogen_ground(wf.data)
    return wf
