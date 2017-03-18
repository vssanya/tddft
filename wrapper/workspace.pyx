from atom cimport Atom
from abs_pot cimport Uabs, uabs_zero
from grid cimport ShGrid, SpGrid
from wavefunc cimport SWavefunc
from field cimport Field
from orbitals cimport SOrbitals
from sphere_harmonics cimport YlmCache


cdef class SKnWorkspace:
    def __cinit__(self, ShGrid grid, int num_threads = -1):
        self.data = sh_workspace_alloc(
            grid.data,
            Uabs,
            num_threads
        )

    def __dealloc__(self):
        if self.data != NULL:
            sh_workspace_free(self.data)

    def prop(self, SWavefunc wf, Atom atom, Field E, double t, double dt):
        sh_workspace_prop(self.data, wf.data, atom._data, E.data, t, dt)

    def prop_img(self, SWavefunc wf, Atom atom, double dt):
        sh_workspace_prop_img(self.data, wf.data, atom._data, dt)


cdef class SOrbsWorkspace:
    def __cinit__(self, ShGrid sh_grid, SpGrid sp_grid, YlmCache ylm_cache, int num_threads=-1):
        self._data = sh_orbs_workspace_alloc(sh_grid.data, sp_grid.data, Uabs, ylm_cache._data, num_threads)

    def __dealloc__(self):
        if self._data != NULL:
            sh_orbs_workspace_free(self._data)

    def prop_img(self, SOrbitals orbs, Atom atom, double dt):
        sh_orbs_workspace_prop_img(self._data, orbs._data, atom._data, dt)

    def prop(self, SOrbitals orbs, Atom atom, Field f, double t, double dt):
        sh_orbs_workspace_prop(self._data, orbs._data, atom._data, f.data, t, dt)
