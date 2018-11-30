from grid cimport ShGrid

import numpy as np

import tdse.utils
if tdse.utils.is_jupyter_notebook():
    import matplotlib.pyplot as plt
    from IPython.core.pylabtools import print_figure

from libc.stdlib cimport free


def test_cos_hump(double x):
    return CosHump.u(x)

def test_pt_hump(double x):
    return PTHump.u(x)

cdef class Uabs:
    def u(self, ShGrid grid, double r):
        return self.cdata.u(grid.data[0], r)

cdef class UabsCache:
    def __cinit__(self, Uabs uabs, ShGrid grid, double[::1] u = None):
        if u is None:
            self.cdata = new cUabsCache(uabs.cdata[0], grid.data[0], NULL)
        else:
            self.cdata = new cUabsCache(uabs.cdata[0], grid.data[0], &u[0])

        self.uabs = uabs
        self.grid = grid

    def __init__(self, Uabs uabs, ShGrid grid, double[::1] u = None):
        pass

    def __dealloc__(self):
        del self.cdata

    @property
    def u(self):
        return np.asarray(<double[:self.grid.Nr]> self.cdata.data)

    def _figure_data(self, format):
        fig, ax = plt.subplots()
        fig.set_size_inches((6,3))

        ax.plot(self.grid.r, self.u)

        ax.set_xlabel('r, (a.u.)')
        ax.set_ylabel('U, (a.u.)')

        ax.set_yscale('log')

        data = print_figure(fig, format)
        plt.close(fig)
        return data

    def _repr_png_(self):
        return self._figure_data('png')

    def write_params(self, params_grp):
        subgrp = params_grp.create_group("uabs")

        subgrp.create_dataset("u", (self.grid.Nr,), dtype='double')[:] = self.u
        self.uabs.write_params(subgrp)

cdef class UabsMultiHump(Uabs):
    def __cinit__(self, double l_min, double l_max, int n = 2, double shift = 0.0):
        self.cdata = <cUabs*> new cUabsMultiHump(l_min, l_max, n, shift)

    def __init__(self, double l_min, double l_max, int n = 2, double shift = 0.0):
        pass

    def write_params(self, params_grp):
        params_grp.attrs['type'] = "MultiHump"

    def l(self, int i):
        return (<cUabsMultiHump*>self.cdata).l[i]

    def getHumpAmplitude(self, int i):
        return (<cUabsMultiHump*>self.cdata).a[i]

    def setHumpAmplitude(self, int i, double value):
        (<cUabsMultiHump*>self.cdata).a[i] = value

    def multHumpAmplitude(self, int i, double value):
        (<cUabsMultiHump*>self.cdata).a[i] *= value

    def getHumpLength(self, int i):
        return (<cUabsMultiHump*>self.cdata).l[i]

    def setHumpLength(self, int i, double value):
        (<cUabsMultiHump*>self.cdata).l[i] = value

cdef class UabsZero(Uabs):
    def __cinit__(self):
        self.cdata = <cUabs*> new cUabsZero()

    def write_params(self, params_grp):
        params_grp.attrs['type'] = "Zero"
