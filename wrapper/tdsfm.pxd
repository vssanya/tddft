from types cimport cdouble
from grid cimport sh_grid_t, sp_grid_t, SpGrid, ShGrid
from field cimport field_t
from sphere_harmonics cimport ylm_cache_t
from wavefunc cimport sh_wavefunc_t


cdef extern from "tdsfm.h":
    ctypedef struct tdsfm_t:
        sp_grid_t* k_grid
        sh_grid_t* r_grid
        int ir
        cdouble* data
        double* jl
        ylm_cache_t* ylm
        double int_A
        double int_A2

    tdsfm_t* tdsfm_new(sp_grid_t * k_grid, sh_grid_t * r_grid, int ir)
    void tdsfm_del(tdsfm_t* tdsfm)
    void tdsfm_calc(tdsfm_t* tdsfm, field_t * field, sh_wavefunc_t * wf, double t, double dt)


cdef class TDSFM:
    cdef tdsfm_t* cdata
    cdef SpGrid k_grid
    cdef ShGrid r_grid
