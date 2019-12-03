from grid cimport Grid1d

"""
cdef class MaxwellWorkspace1D:
    def __cinit__(self, Grid1d grid):
        self.cdata = new cWorkspace1D(grid.cdata)

    def __init__(self, Grid1d grid):
        pass

    @property
    def E(self):
        cdef double[::1] res = <double[:self.cdata.grid.n]>self.cdata.E
        return res

    @property
    def H(self):
        cdef double[::1] res = <double[:self.cdata.grid.n]>self.cdata.H
        return res

    @property
    def D(self):
        cdef double[::1] res = <double[:self.cdata.grid.n]>self.cdata.D
        return res

    def prop(self, double dt, double[::1] eps = None):
        if eps is None:
            self.cdata.prop(dt)
        else:
            self.cdata.prop(dt, &eps[0])
"""
