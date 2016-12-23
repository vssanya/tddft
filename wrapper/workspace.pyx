from hydrogen cimport hydrogen_U
from abs_pot cimport Uabs
from grid cimport SGrid
from wavefunc cimport SWavefunc
from field cimport Field

cdef class SKnWorkspace:
    def __cinit__(self, SGrid grid, double dt):
        self.data = sphere_kn_workspace_alloc(
            grid.data,
            dt, hydrogen_U, Uabs
        )

    def __dealloc__(self):
        sphere_kn_workspace_free(self.data)

    def prop(self, SWavefunc wf, Field E, double t):
        sphere_kn_workspace_prop(self.data, wf.data, E.data, t)
