from grid cimport sh_grid_t

ctypedef double complex complex_t

cdef extern from "types.h":
    ctypedef struct cdouble:
        double real
        double imag
    ctypedef double (*sh_f)(sh_grid_t* grid, int ir, int l, int m)
