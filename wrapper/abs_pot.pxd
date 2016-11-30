from grid cimport sphere_grid_t

cdef extern from "abs_pot.h":
    double Uabs(double r, sphere_grid_t* grid)
