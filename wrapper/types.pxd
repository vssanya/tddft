ctypedef double complex cdouble

cdef extern from "types.h":
    ctypedef double (*sphere_pot_t)(double r)
    ctypedef double (*field_t)(double t)
