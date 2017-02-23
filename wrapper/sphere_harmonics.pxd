from integrate cimport func_2d_t
from grid cimport sp_grid_t

cdef extern from "sphere_harmonics.h":
    int pow_minus_one(int p)
    double clebsch_gordan_coef(int j1, int m1, int j2, int m2, int J, int M)
    double y3(int l1, int m1, int l2, int m2, int L, int M)

    void   ylm_cache_init(int l_max, sp_grid_t* grid);
    void   ylm_cache_deinit();
    double ylm(int l, int m, int c)

    void sh_series(func_2d_t func, int l, int m, sp_grid_t* grid, double* series)
