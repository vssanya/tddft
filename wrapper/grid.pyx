import numpy as np
from libc.stdlib cimport free


cdef class Grid1d:
    def __cinit__(self, int N, double d):
        self.cdata = cGrid1d(N, d)

    def __init__(self, int N, double d):
        pass

    @property
    def d(self):
        return self.cdata.d

    @property
    def N(self):
        return self.cdata.n


cdef class CtGrid:
    def __cinit__(self, int Nx, int Ny, double Xmax, double Ymax):
        self.cdata = new cCtGrid([Nx, Ny], Xmax, Ymax)

    def __init__(self, int Nx, int Ny, double Xmax, double Ymax):
        pass

    def __dealloc__(self):
        del self.cdata


cdef class SpGrid2d:
    def __cinit__(self, int Nr, int Nc, double Rmax):
        self.cdata = new cSpGrid2d([Nr, Nc], Rmax)

    def __init__(self, int Nr, int Nc, double Rmax):
        pass

    def __dealloc__(self):
        del self.cdata

cdef class ShGrid:
    def __cinit__(self, int Nr, int Nl, double r_max):
        self.data = new cShGrid([Nr, Nl], r_max)

    def __init__(self, int Nr, int Nl, double r_max):
        pass

    def __dealloc__(self):
        del self.data

    def _repr_latex_(self):
        return "ShGrid: $N_r = {}$, $N_l = {}$, $dr = {}$ (a.u.)".format(self.data.n[0], self.data.n[1], self.data.d[0])

    @property
    def shape(self):
        return (self.data.n[1], self.data.n[0])

    @property
    def Nl(self):
        return self.data.n[1]

    @property
    def Nr(self):
        return self.data.n[0]

    @property
    def Rmax(self):
        return self.data.Rmax()

    @property
    def r(self):
        return np.linspace(self.data.d[0], self.data.d[0]*self.data.n[0], self.data.n[0])

cdef class SpGrid:
    def __cinit__(self, int Nr, int Nc, int Np, double r_max):
        self.data = new cSpGrid([Nr, Nc, Np], r_max)

    def __init__(self, int Nr, int Nc, int Np, double r_max):
        pass

    def __dealloc__(self):
        del self.data

    @property
    def r(self):
        return np.linspace(self.data.d[0], self.data.d[0]*self.data.n[0], self.data.n[0])

    @property
    def shape(self):
        return (self.data.n[1], self.data.n[0])
