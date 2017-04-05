from grid cimport sh_grid_t

cdef extern from "abs_pot.h":
    ctypedef struct uabs_sh_t:
        double (*func)(void* self, sh_grid_t* grid, int ir, int il, int im)

    ctypedef struct uabs_multi_hump_t:
        double (*func)(void* self, sh_grid_t* grid, int ir, int il, int im)
        double l[3]
        double u[3]
        double r0[3]

    double uabs(sh_grid_t* grid, int ir, int il, int im)
    double uabs_zero(sh_grid_t* grid, int ir, int il, int im)
    double mask_core(sh_grid_t* grid, int ir, int il, int im)

    double uabs_get(uabs_sh_t* self, sh_grid_t* grid, int ir, int il, int im)

    uabs_multi_hump_t* uabs_multi_hump_new(double lambda_min, double lambda_max)
    double uabs_multi_hump_func(uabs_multi_hump_t* self, sh_grid_t* grid, int ir, int il, int im)

    uabs_sh_t uabs_zero;

cdef class Uabs:
    cdef uabs_sh_t* cdata
    cdef bint _dealloc
