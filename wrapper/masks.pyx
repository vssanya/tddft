import tdse.utils
from grid cimport ShGrid, ShNeGrid

if tdse.utils.is_jupyter_notebook():
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.core.pylabtools import print_figure

cdef class CoreMask:
    def __cinit__(self, ShGrid grid, double r_core, double dr):
        self.cdata = new cCoreMask(<cShGrid*>grid.data, r_core, dr)

    def __init__(self, ShGrid grid, double r_core, double dr):
        self.grid = grid

    def __dealloc__(self):
        del self.cdata

    def write_params(self, params_grp):
        params_grp.attrs['mask_type'] = "CoreMask"
        params_grp.attrs['mask_r_core'] = self.cdata.r_core
        params_grp.attrs['mask_dr'] = self.cdata.dr

    def __call__(self, ir):
        return self.cdata[0](ir, 0, 0)

    def _repr_png_(self):
        return self._figure_data('png')

    def _figure_data(self, format):
        cdef int i

        fig, ax = plt.subplots()
        fig.set_size_inches((6,3))

        r = self.grid.r
        u = np.zeros(r.size)

        for i in range(u.size):
            u[i] = self.cdata[0](i, 0, 0)

        ax.plot(r, u)

        ax.set_xlabel('r, (a.u.)')
        ax.set_ylabel('U, (a.u.)')

        data = print_figure(fig, format)
        plt.close(fig)
        return data
