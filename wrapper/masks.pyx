import tdse.utils
from grid cimport ShGrid

if tdse.utils.is_jupyter_notebook():
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.core.pylabtools import print_figure


cdef class CoreMask:
    def __cinit__(self, double r_core, double dr):
        self.cdata = new cCoreMask(r_core, dr)

    def __init__(self, double r_core, double dr):
        pass

    def __dealloc__(self):
        del self.cdata

    def write_params(self, params_grp):
        params_grp.attrs['mask_type'] = "CoreMask"
        params_grp.attrs['mask_r_core'] = self.cdata.r_core
        params_grp.attrs['mask_dr'] = self.cdata.dr

    def __call__(self, ShGrid grid, ir):
        return self.cdata[0](grid.data, ir, 0, 0)

    def _repr_png_(self):
        return self._figure_data('png')

    def _figure_data(self, format):
        cdef int i

        fig, ax = plt.subplots()
        fig.set_size_inches((6,3))

        r_max = self.cdata.r_core+2*self.cdata.dr
        grid = ShGrid(int(r_max / self.cdata.dr * 100), 1, r_max)
        r = grid.r
        u = np.zeros(r.size)

        for i in range(u.size):
            u[i] = self.cdata[0](grid.data, i, 0, 0)

        ax.plot(r, u)

        ax.set_xlabel('r, (a.u.)')
        ax.set_ylabel('U, (a.u.)')

        data = print_figure(fig, format)
        plt.close(fig)
        return data
