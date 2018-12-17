from grid cimport cShGrid, ShGrid
from libcpp.vector cimport vector

cdef extern from "abs_pot.h":
    double uabs(cShGrid* grid, int ir, int il, int im)
    double mask_core(cShGrid* grid, int ir, int il, int im)

    cdef cppclass cUabs "Uabs":
        double u(cShGrid& grid, double r)
        double getWidth()
        void calcAbs(int N, double* l, double* res)

    cdef cppclass cUabsZero "UabsZero":
        pass

    cdef cppclass cHump "Hump":
        double u(double)

    cdef cHump CosHump
    cdef cHump PTHump

    cdef cppclass cUabsMultiHump "UabsMultiHump":
        cUabsMultiHump(double l_min, double l_max, vector[cHump], double shift)
        cUabsMultiHump(double l_min, double l_max, int n, double shift)

        double getHumpAmplitude(int i)
        void setHumpAmplitude(int i, double value)

        vector[double] l
        vector[double] a

    cdef cppclass cUabsCache "UabsCache":
        cUabsCache(cUabs& uabs, cShGrid* grid, double* u)
        double* data

cdef class Uabs:
    cdef cUabs* cdata

cdef class UabsCache:
    cdef cUabsCache* cdata
    cdef public Uabs uabs
    cdef dict __dict__ #cdef public ShGrid grid

cdef class UabsNeCache(UabsCache):
    pass

cdef class UabsZero(Uabs):
    pass

cdef class UabsMultiHump(Uabs):
    pass
