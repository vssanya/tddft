from grid cimport cShGrid, ShGrid

cdef extern from "abs_pot.h":
    double uabs(cShGrid* grid, int ir, int il, int im)
    double mask_core(cShGrid* grid, int ir, int il, int im)

    cdef cppclass cUabs "Uabs":
        double u(cShGrid& grid, double r)

    cdef cppclass cUabsZero "UabsZero":
        pass

    cdef cppclass cUabsMultiHump "UabsMultiHump":
        cUabsMultiHump(double l_min, double l_max)

    cdef cppclass cUabsCache "UabsCache":
        cUabsCache(cUabs& uabs, cShGrid& grid, double* u)
        double* data

cdef class Uabs:
    cdef cUabs* cdata

cdef class UabsCache:
    cdef cUabsCache* cdata
    cdef public ShGrid grid
    cdef public Uabs uabs

cdef class UabsZero(Uabs):
    pass

cdef class UabsMultiHump(Uabs):
    pass
