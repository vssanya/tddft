cdef extern from "sphere_grid.h":
    ctypedef struct sphere_grid_t:
        int Nr, Nl
        double dr

cdef class SGrid:
    cdef sphere_grid_t data
