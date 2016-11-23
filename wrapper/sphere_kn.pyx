from hydrogen cimport hydrogen_U
from abs_pot cimport Uabs
from sphere_grid cimport SGrid

cdef class SKnWorkspace:
    def __cinit__(self, SGrid grid, double dt):
        self.data = sphere_kn_workspace_alloc(
            <sphere_grid_t*> &grid.data,
            dt, hydrogen_U, Uabs
        )

    def __dealloc__(self):
        sphere_kn_workspace_free(self.data)
