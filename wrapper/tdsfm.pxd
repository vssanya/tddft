from types cimport cdouble
from grid cimport cShGrid, cSpGrid, SpGrid, ShGrid
from field cimport field_t
from sphere_harmonics cimport cYlmCache, YlmCache, cJlCache
from wavefunc cimport cShWavefunc

from libcpp cimport bool

cimport numpy as np


cdef extern from "tdsfm.h":
    cdef cppclass TDSFM_Base:
        cSpGrid* k_grid
        cShGrid* r_grid

        int ir

        cdouble* data

        cSpGrid* jl_grid
        cJlCache* jl

        cSpGrid* ylm_grid
        cYlmCache* ylm

        double int_A
        double int_A2

        void init_cache()

        void calc(field_t* field, cShWavefunc& wf, double t, double dt, double mask)
        void calc_inner(field_t* field, cShWavefunc& wf, double t, int ir_min, int ir_max, int l_min, int l_max)

        double pz()
        double norm()
        void calc_norm_k(cShWavefunc& wf, int ir_min, int ir_max, int l_min, int l_max)

    cdef cppclass TDSFM_E:
        TDSFM_E(cSpGrid* k_grid, cShGrid* r_grid, double A_max, int ir, bool init_cache)

    cdef cppclass TDSFM_A:
        TDSFM_A(cSpGrid* k_grid, cShGrid* r_grid, int ir, bool init_cache)


cdef class TDSFM:
    cdef TDSFM_Base* cdata
    cdef SpGrid k_grid
    cdef ShGrid r_grid

cdef class TDSFM_LG(TDSFM):
    pass

cdef class TDSFM_VG(TDSFM):
    pass
