from grid cimport cShGrid, cShNeGrid, ShGrid, ShNeGrid, cSpGrid2d, SpGrid2d

cdef extern from "masks.h":
    cdef cppclass CoreMask[Grid]:
        double r_core
        double dr

        CoreMask(Grid* grid, double r_core, double dr)
        double operator()(int ir, int il)
        double operator()(int ir, int il, int im)
        double* getGPUData()

cdef class ShCoreMask:
    cdef CoreMask[cShGrid]* cdata
    cdef ShGrid grid

cdef class ShNeCoreMask:
    cdef CoreMask[cShNeGrid]* cdata
    cdef ShNeGrid grid

cdef class SpCoreMask:
    cdef CoreMask[cSpGrid2d]* cdata
    cdef SpGrid2d grid
