from grid cimport sh_grid_t

cdef extern from "abs_pot.h":
    double Uabs(double r, sh_grid_t* grid)

