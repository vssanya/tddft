cdef class SGrid:
    def __cinit__(self, int Nr, int Nl, double r_max):
        self.data = sh_grid_new([Nr, Nl], r_max)

    def __dealloc__(self):
        sh_grid_del(self.data)

cdef class SpGrid:
    def __cinit__(self, int Nr, int Nc, int Np, double r_max):
        self.data = sp_grid_new([Nr, Nc, Np], r_max)

    def __dealloc__(self):
        sp_grid_del(self.data)
