from grid cimport cShGrid

cdef extern from "masks.h":
    cdef cppclass cCoreMask "CoreMask":
        double r_core
        double dr

        cCoreMask(cShGrid* grid, double r_core, double dr)
        double operator()(int ir, int il, int im)

cdef class CoreMask:
    cdef cCoreMask* cdata
