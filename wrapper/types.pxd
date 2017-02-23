from grid cimport sh_grid_t

ctypedef double complex cdouble

cdef extern from "types.h":
    ctypedef double (*sh_f)(sh_grid_t* grid, int ir, int l, int m)
