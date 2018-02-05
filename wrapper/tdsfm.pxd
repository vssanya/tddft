from types cimport cdouble
from grid cimport sh_grid_t, sp_grid_t, SpGrid, ShGrid
from field cimport field_t
from sphere_harmonics cimport ylm_cache_t, YlmCache
from wavefunc cimport sh_wavefunc_t

from libcpp cimport bool

cimport numpy as np


cdef extern from "tdsfm.h":
    cdef cppclass TDSFM_Base:
        void calc(field_t* field, sh_wavefunc_t& wf, double t, double dt)
        void calc_inner(field_t* field, sh_wavefunc_t& wf, double t, int ir_min, int ir_max)
        double pz()

    cdef cppclass TDSFM_E:
        tdsfm_t(sp_grid_t* k_grid, sh_grid_t* r_grid, double A_max, int ir, bool init_cache)

    cdef cppclass TDSFM_A:
        tdsfm_t(sp_grid_t* k_grid, sh_grid_t* r_grid, int ir, bool init_cache)


cdef class TDSFM:
    cdef TDSFM_Base* cdata
    cdef SpGrid k_grid
    cdef ShGrid r_grid

cdef class TDSFM_LG(TDSFM_Base):
    pass

cdef class TDSFM_VG:
    pass
