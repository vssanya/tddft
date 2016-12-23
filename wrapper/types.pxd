from grid cimport sh_grid_t

ctypedef double complex cdouble

cdef extern from "types.h":
    ctypedef double (*sphere_pot_t)(double r)
    ctypedef double (*sphere_pot_abs_t)(double r, sh_grid_t* grid)
