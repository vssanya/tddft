from grid cimport cShGrid
from libcpp.functional cimport function

ctypedef double complex complex_t

cdef extern from "types.h":
    ctypedef struct cdouble:
        double real
        double imag
    ctypedef function[double(cShGrid* grid, int ir, int il, int m)] sh_f
