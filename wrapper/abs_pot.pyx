from grid cimport ShGrid

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import print_figure

from libc.stdlib cimport free


cdef class Uabs:
    def __dealloc__(self):
        if self._dealloc:
            free(self.cdata)

    def __call__(self, ShGrid grid, int ir=0, int il=0, int im=0):
        return uabs_get(self.cdata, grid.data, ir, il, im)

    def _figure_data(self, format, ShGrid grid = ShGrid(1000, 1, 100)):
        r = grid.get_r()
        U = np.ndarray(r.shape)
        for i in range(r.size):
            U[i] = self(grid, i)

        fig, ax = plt.subplots()
        fig.set_size_inches((6,3))

        ax.plot(r, U)

        ax.set_xlabel('r, (a.u.)')
        ax.set_ylabel('U, (a.u.)')

        ax.set_yscale('log')

        data = print_figure(fig, format)
        plt.close(fig)
        return data

    def _repr_png_(self):
        return self._figure_data('png')

cdef class UabsMultiHump(Uabs):
    def __cinit__(self, double lambda_min, double lambda_max):
        self.cdata = <uabs_sh_t*>uabs_multi_hump_new(lambda_min, lambda_max)
        self._dealloc = True

    def __init__(self, double lambda_min, double lambda_max):
        pass

cdef class UabsZero(Uabs):
    def __cinit__(self):
        self.cdata = &uabs_zero
        self._dealloc = False
