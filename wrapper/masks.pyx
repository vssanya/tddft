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
