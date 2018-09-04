cdef extern from "masks.h":
    cdef cppclass cCoreMask "CoreMask":
        double r_core
        double dr

        cCoreMask(double r_core, double dr)

cdef class CoreMask:
    cdef cCoreMask* cdata
