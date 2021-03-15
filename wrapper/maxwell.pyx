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
