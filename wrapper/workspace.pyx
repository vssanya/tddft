from atom cimport hydrogen_sh_u, argon_sh_u, neon_sh_u
from abs_pot cimport Uabs, uabs_zero
from grid cimport SGrid
from wavefunc cimport SWavefunc
from field cimport Field
from orbitals cimport SOrbitals

cdef class SKnWorkspace:
    def __cinit__(self, SGrid grid, double dt):
        self.data = sphere_kn_workspace_alloc(
            grid.data,
            dt, hydrogen_sh_u, Uabs
        )

    def __dealloc__(self):
        sphere_kn_workspace_free(self.data)

    def prop(self, SWavefunc wf, Field E, double t):
        sphere_kn_workspace_prop(self.data, wf.data, E.data, t)

    def prop_img(self, SWavefunc wf):
        sphere_kn_workspace_prop_img(self.data, wf.data)

cdef class SOrbsWorkspace:
    def __cinit__(self, SGrid grid, double dt):
        self._data = sphere_kn_orbs_workspace_alloc( grid.data, dt, neon_sh_u, Uabs )

    def __dealloc__(self):
        sphere_kn_orbs_workspace_free(self._data)

    def prop_img(self, SOrbitals orbs):
        sphere_kn_orbs_workspace_prop_img(self._data, orbs._data)
