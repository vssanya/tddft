import numpy as np
from libc.stdlib cimport free


cdef class CtGrid:
    def __cinit__(self, int Nx, int Ny, double Xmax, double Ymax):
        self.cdata = ct_grid_new([Nx, Ny], Xmax, Ymax)

    def __init__(self, int Nx, int Ny, double Xmax, double Ymax):
        pass

    def __dealloc__(self):
        free(self.cdata)

    def size(self):
        grid2_size(self.cdata)


cdef class Sp2Grid:
    def __cinit__(self, int Nr, int Nc, double Rmax):
        self.cdata = sp2_grid_new([Nr, Nc], Rmax)

    def __init__(self, int Nr, int Nc, double Rmax):
        pass

    def __dealloc__(self):
        free(self.cdata)

    def size(self):
        grid2_size(self.cdata)


cdef class ShGrid:
    def __cinit__(self, int Nr, int Nl, double r_max):
        self.data = sh_grid_new([Nr, Nl], r_max)

    def __init__(self, int Nr, int Nl, double r_max):
        pass

    def __dealloc__(self):
        sh_grid_del(self.data)

    def _repr_latex_(self):
        return "ShGrid: $N_r = {}$, $N_l = {}$, $dr = {}$ (a.u.)".format(self.data.n[0], self.data.n[1], self.data.d[0])

    @property
    def shape(self):
        return (self.data.n[1], self.data.n[0])

    def get_r(self):
        return np.linspace(self.data.d[0], self.data.d[0]*self.data.n[0], self.data.n[0])

cdef class SpGrid:
    def __cinit__(self, int Nr, int Nc, int Np, double r_max):
        self.data = sp_grid_new([Nr, Nc, Np], r_max)

    def __init__(self, int Nr, int Nc, int Np, double r_max):
        pass

    def __dealloc__(self):
        sp_grid_del(self.data)

    def get_r(self):
        return np.linspace(self.data.d[0], self.data.d[0]*self.data.n[0], self.data.n[0])

    @property
    def shape(self):
        return (self.data.n[1], self.data.n[0])
