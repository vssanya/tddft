from types cimport cdouble
from grid cimport sh_grid_t, sp_grid_t, SpGrid, ShGrid
from field cimport field_t
from sphere_harmonics cimport ylm_cache_t, YlmCache
from wavefunc cimport sh_wavefunc_t

from libcpp cimport bool

cimport numpy as np


cdef extern from "tdsfm.h":
    cdef cppclass tdsfm_t:
        sp_grid_t* k_grid;
        sh_grid_t* r_grid;
        int ir;
        cdouble* data

        tdsfm_t(sp_grid_t* k_grid, sh_grid_t* r_grid, double A_max, int ir, bool init_cache)
        void init_cache()
        void calc(field_t* field, sh_wavefunc_t& wf, double t, double dt)
        void calc_inner(field_t* field, sh_wavefunc_t& wf, double t, int ir_min, int ir_max)
        double pz()


cdef class TDSFM:
    cdef tdsfm_t* cdata
    cdef SpGrid k_grid
    cdef ShGrid r_grid
