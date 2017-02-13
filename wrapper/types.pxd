from grid cimport sh_grid_t

ctypedef double complex cdouble

cdef extern from "types.h":
    ctypedef double (*sphere_pot_t)(sh_grid_t* grid, int ir, int l, int m)
