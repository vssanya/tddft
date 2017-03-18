from grid cimport sh_grid_t

cdef extern from "abs_pot.h":
    double Uabs(sh_grid_t* grid, int ir, int il, int im)
    double uabs_zero(sh_grid_t* grid, int ir, int il, int im)
    double mask_core(sh_grid_t* grid, int ir, int il, int im)
