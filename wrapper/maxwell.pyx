import numpy as np
from grid cimport Grid1d


cdef class MaxwellWorkspace1D:
    def __cinit__(self, Grid1d grid):
        self.cdata = new cWorkspace1D(grid.cdata)

    def __init__(self, Grid1d grid):
        pass

    @property
    def E(self):
        cdef double[::1] res = <double[:self.cdata.grid.n]>self.cdata.E.data
        return np.asarray(res)

    @property
    def H(self):
        cdef double[::1] res = <double[:self.cdata.grid.n]>self.cdata.H.data
        return res

    @property
    def D(self):
        cdef double[::1] res = <double[:self.cdata.grid.n]>self.cdata.D.data
        return res

    def prop(self, double ksi = 0.9, dt = None, double[::1] eps = None, double[::1] pol = None):
        if dt is None:
            dt = self.get_dt(ksi)

        if eps is not None:
            self.cdata.prop(<double> dt, &eps[0])
        elif pol is not None:
            self.cdata.prop_pol(<double> dt, &pol[0])
        else:
            self.cdata.prop(<double> dt)

    def move_center_window_to_max_E(self):
        return self.cdata.move_center_window_to_max_E()

    def get_dt(self, double ksi):
        return ksi/C_au*self.cdata.grid.d

cdef class MaxwellWorkspaceCyl1D:
    def __cinit__(self, Grid1d grid):
        self.cdata = new cWorkspaceCyl1D(grid.cdata)

    def __init__(self, Grid1d grid):
        pass

    @property
    def Er(self):
        cdef double[::1] res = <double[:self.cdata.grid.n]>self.cdata.Er.data
        return np.asarray(res)

    @property
    def Ephi(self):
        cdef double[::1] res = <double[:self.cdata.grid.n]>self.cdata.Ephi.data
        return np.asarray(res)

    @property
    def Hz(self):
        cdef double[::1] res = <double[:self.cdata.grid.n]>self.cdata.Hz.data
        return res

    def prop(self, double dt, double[::1] j):
        self.cdata.prop(dt, &j[0])

    def get_dt(self, double ksi):
        return ksi/C_au*self.cdata.grid.d


cdef class MaxwellWorkspace3D:
    def __cinit__(self, Grid3d grid):
        self.cdata = new cWorkspace3D(grid.cdata)

    def __init__(self, Grid3d grid):
        pass

    @property
    def Ex(self):
        cdef double[:, :, ::1] res = <double[:self.cdata.grid.n[0],:self.cdata.grid.n[1],:self.cdata.grid.n[2]]>self.cdata.Ex.data
        return np.asarray(res)

    @property
    def Ey(self):
        cdef double[:, :, ::1] res = <double[:self.cdata.grid.n[0],:self.cdata.grid.n[1],:self.cdata.grid.n[2]]>self.cdata.Ey.data
        return np.asarray(res)

    @property
    def Ez(self):
        cdef double[:, :, ::1] res = <double[:self.cdata.grid.n[0],:self.cdata.grid.n[1],:self.cdata.grid.n[2]]>self.cdata.Ez.data
        return np.asarray(res)

    @property
    def Hx(self):
        cdef double[:, :, ::1] res = <double[:self.cdata.grid.n[0],:self.cdata.grid.n[1],:self.cdata.grid.n[2]]>self.cdata.Hx.data
        return np.asarray(res)

    @property
    def Hy(self):
        cdef double[:, :, ::1] res = <double[:self.cdata.grid.n[0],:self.cdata.grid.n[1],:self.cdata.grid.n[2]]>self.cdata.Hy.data
        return np.asarray(res)

    @property
    def Hz(self):
        cdef double[:, :, ::1] res = <double[:self.cdata.grid.n[0],:self.cdata.grid.n[1],:self.cdata.grid.n[2]]>self.cdata.Hz.data
        return np.asarray(res)

    def prop(self, double dt):
        self.cdata.prop(dt)

    def get_dt(self, int index, double ksi):
        return ksi/C_au*self.cdata.grid.d[index]
