cdef class ShGrid:
    def __cinit__(self, int Nr, int Nl, double r_max):
        self.data = sh_grid_new([Nr, Nl], r_max)

    def __dealloc__(self):
        sh_grid_del(self.data)

    @property
    def shape(self):
        return (self.data.n[1], self.data.n[0])

cdef class SpGrid:
    def __cinit__(self, int Nr, int Nc, int Np, double r_max):
        self.data = sp_grid_new([Nr, Nc, Np], r_max)

    def __dealloc__(self):
        sp_grid_del(self.data)

    @property
    def shape(self):
        return (self.data.n[1], self.data.n[0])
