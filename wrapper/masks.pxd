from grid cimport cShGrid, cShNeGrid, ShGrid, ShNeGrid

cdef extern from "masks.h":
    cdef cppclass CoreMask[Grid]:
        double r_core
        double dr

        CoreMask(Grid* grid, double r_core, double dr)
        double operator()(int ir, int il, int im)

cdef class ShCoreMask:
    cdef CoreMask[cShGrid]* cdata
    cdef ShGrid grid

cdef class ShNeCoreMask:
    cdef CoreMask[cShNeGrid]* cdata
    cdef ShNeGrid grid
