from grid cimport cShGrid

ctypedef double complex complex_t

cdef extern from "types.h":
    ctypedef struct cdouble:
        double real
        double imag
    ctypedef double (*sh_f)(cShGrid* grid, int ir, int l, int m)
