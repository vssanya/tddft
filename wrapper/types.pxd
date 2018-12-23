from grid cimport cShGrid
from libcpp.functional cimport function

ctypedef double complex complex_t
ctypedef function[double(int ir, int il, int m)] sh_f

cdef extern from "types.h":
    ctypedef struct cdouble:
        double real
        double imag
