from types cimport cdouble
from grid cimport sh_grid_t, sp_grid_t, SpGrid, ShGrid
from field cimport field_t
from sphere_harmonics cimport ylm_cache_t, YlmCache, jl_cache_t
from wavefunc cimport sh_wavefunc_t

from libcpp cimport bool

cimport numpy as np


cdef extern from "tdsfm.h":
    cdef cppclass TDSFM_Base:
        sp_grid_t* k_grid
        sh_grid_t* r_grid

        int ir

        cdouble* data

        sp_grid_t* jl_grid
        jl_cache_t* jl

        sp_grid_t* ylm_grid
        ylm_cache_t* ylm

        double int_A
        double int_A2

        void init_cache()

        void calc(field_t* field, sh_wavefunc_t& wf, double t, double dt)
        void calc_inner(field_t* field, sh_wavefunc_t& wf, double t, int ir_min, int ir_max)
        double pz()

    cdef cppclass TDSFM_E:
        TDSFM_E(sp_grid_t* k_grid, sh_grid_t* r_grid, double A_max, int ir, bool init_cache)

    cdef cppclass TDSFM_A:
        TDSFM_A(sp_grid_t* k_grid, sh_grid_t* r_grid, int ir, bool init_cache)


cdef class TDSFM:
    cdef TDSFM_Base* cdata
    cdef SpGrid k_grid
    cdef ShGrid r_grid

cdef class TDSFM_LG(TDSFM):
    pass

cdef class TDSFM_VG(TDSFM):
    pass
