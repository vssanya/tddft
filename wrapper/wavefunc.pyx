from grid cimport SGrid

cdef class SWavefunc:
    def __cinit__(self, SGrid grid, int m=0):
        self.data = sphere_wavefunc_alloc(&grid.data, m)

    def __dealloc__(self):
        sphere_wavefunc_free(self.data)

    def norm(self):
        sphere_wavefunc_norm(self.data)

    def normalize(self):
        sphere_wavefunc_normalize(self.data)

    def z(self):
        sphere_wavefunc_z(self.data)
