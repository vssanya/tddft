import numpy as np
cimport numpy as np

from libc.stdlib cimport free


cdef class Range:
    def __cinit__(self, int start, int end):
        self.cdata = cRange(start, end)

    def __init__(self, int start, int end):
        pass

    @property
    def start(self) -> int:
        return self.cdata.start

    @property
    def end(self) -> int:
        return self.cdata.end


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

    def createGridWith(self, Nl):
        return ShGrid(self.Nr, Nl, self.Rmax)

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
    def dr(self):
        return self.data.d[0]

    @property
    def r(self):
        return np.linspace(self.data.d[0], self.data.d[0]*self.data.n[0], self.data.n[0])

cdef class ShNeGrid:
    def __cinit__(self, double Rmin, double Rmax, double Ra, double dr_max, int Nl):
        self.Rmin = Rmin
        self.Ra = Ra
        self.data = new cShNeGrid(Rmin, Rmax, Ra, dr_max, Nl)

    def __init__(self, double Rmin, double Rmax, double Ra, double dr_max, int Nl):
        pass

    def __dealloc__(self):
        del self.data

    def _repr_latex_(self):
        return "ShNeGrid: $N_r = {}$, $N_l = {}$, $dr = {}$ (a.u.)".format(self.data.n[0], self.data.n[1], self.data.d[0])

    def createGridWith(self, Nl):
        return ShNeGrid(self.Rmin, self.Rmax, self.Ra, self.dr, Nl)

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
    def dr(self):
        return self.data.d[0]

    @property
    def r(self):
        cdef int i
        cdef np.ndarray[np.double_t, ndim=1] res = np.zeros(self.data.n[0])

        for i in range(self.data.n[0]):
            res[i] = self.data.r(i)

        return res
cdef class ShNeGrid3D:
    def __cinit__(self, double Rmin, double Rmax, double Ra, double dr_max, int Nl):
        self.Rmin = Rmin
        self.Ra = Ra
        self.data = new cShNeGrid3D(Rmin, Rmax, Ra, dr_max, Nl)

    def __init__(self, double Rmin, double Rmax, double Ra, double dr_max, int Nl):
        pass

    def __dealloc__(self):
        del self.data

    def _repr_latex_(self):
        return "ShNeGrid3D: $N_r = {}$, $N_l = {}$, $dr = {}$ (a.u.)".format(self.data.n[0], self.data.n[1], self.data.d[0])

    def createGridWith(self, Nl):
        return ShNeGrid3D(self.Rmin, self.Rmax, self.Ra, self.dr, Nl)

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
    def dr(self):
        return self.data.d[0]

    @property
    def r(self):
        cdef int i
        cdef np.ndarray[np.double_t, ndim=1] res = np.zeros(self.data.n[0])

        for i in range(self.data.n[0]):
            res[i] = self.data.r(i)

        return res

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
    def Nr(self):
        return self.data.n[0]

    @property
    def Nc(self):
        return self.data.n[1]

    @property
    def shape(self):
        return (self.data.n[1], self.data.n[0])
