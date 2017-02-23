from atom cimport Atom
from abs_pot cimport Uabs, uabs_zero
from grid cimport SGrid
from wavefunc cimport SWavefunc
from field cimport Field
from orbitals cimport SOrbitals

cdef class SKnWorkspace:
    def __cinit__(self, SGrid grid, Atom atom):
        self.data = sh_workspace_alloc(
            grid.data,
            atom._data.u, Uabs
        )

    def __dealloc__(self):
        sh_workspace_free(self.data)

    def prop(self, SWavefunc wf, Field E, double t, double dt):
        sh_workspace_prop(self.data, wf.data, E.data, t, dt)

    def prop_img(self, SWavefunc wf, double dt):
        sh_workspace_prop_img(self.data, wf.data, dt)

cdef class SOrbsWorkspace:
    def __cinit__(self, SGrid grid, Atom atom):
        self._data = sh_orbs_workspace_alloc(grid.data, atom._data.u, Uabs)

    def __dealloc__(self):
        if self._data != NULL:
            sh_orbs_workspace_free(self._data)

    def prop_img(self, SOrbitals orbs, double dt):
        sh_orbs_workspace_prop_img(self._data, orbs._data, dt)
