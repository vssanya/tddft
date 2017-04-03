from grid cimport ShGrid
from libc.stdlib cimport free

cdef class Uabs:
    def __dealloc__(self):
        free(self.cdata)

    def __call__(self, ShGrid grid, int ir=0, int il=0, int im=0):
        return uabs_get(self.cdata, grid.data, ir, il, im)

cdef class UabsMultiHump(Uabs):
    def __cinit__(self, double lambda_min, double lambda_max):
        self.cdata = <uabs_sh_t*>uabs_multi_hump_new(lambda_min, lambda_max)
