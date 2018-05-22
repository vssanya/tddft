from grid cimport ShGrid

import numpy as np

import tdse.utils
if tdse.utils.is_jupyter_notebook():
    import matplotlib.pyplot as plt
    from IPython.core.pylabtools import print_figure

from libc.stdlib cimport free


cdef class Uabs:
    pass

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

cdef class UabsMultiHump(Uabs):
    def __cinit__(self, double l_min, double l_max):
        self.cdata = <cUabs*> new cUabsMultiHump(l_min, l_max)

    def __init__(self, double l_min, double l_max):
        pass

cdef class UabsZero(Uabs):
    def __cinit__(self):
        self.cdata = <cUabs*> new cUabsZero()
