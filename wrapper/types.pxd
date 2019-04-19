from grid cimport cShGrid
from libcpp.functional cimport function

ctypedef double complex complex_t
ctypedef function[double(int ir, int il, int m)] sh_f

cdef extern from "types.h":
    ctypedef struct cdouble:
        double real
        double imag

cdef extern from "optional" namespace "std":
    cdef cppclass nullopt_t:
        pass

    cdef nullopt_t none

    cdef cppclass optional[T]:
        optional()
        optional(T& data)
        optional(nullopt_t n)
