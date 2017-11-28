from integrate cimport func_2d_t
from grid cimport sp_grid_t

cdef extern from "sphere_harmonics.h":
    ctypedef struct ylm_cache_t:
        double* data
        int size
        int l_max
        sp_grid_t* grid

    int pow_minus_one(int p)
    double clebsch_gordan_coef(int j1, int m1, int j2, int m2, int J, int M)
    double y3(int l1, int m1, int l2, int m2, int L, int M)

    ylm_cache_t* ylm_cache_new(int l_max, sp_grid_t* grid)
    void ylm_cache_del(ylm_cache_t* ylm_cache)
    double ylm_cache_get(ylm_cache_t* cache, int l, int m, int ic)
    double ylm_cache_calc(ylm_cache_t* cache, int l, int m, double c)

    void sh_series(func_2d_t func, int l, int m, sp_grid_t* grid, double* series, ylm_cache_t* ylm_cache);

cdef class YlmCache:
    cdef ylm_cache_t* cdata
