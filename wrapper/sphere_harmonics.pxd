from integrate cimport func_2d_t
from grid cimport cSpGrid2d, cSpGrid
from array cimport ArraySp2D

cdef extern from "sphere_harmonics.h":
    cdef cppclass cJlCache "JlCache":
        cJlCache(cSpGrid grid, int l_max)
        double operator()(int ir, int il)
        double operator()(double r, int il)
        @staticmethod
        double calc(double r, int il)

    cdef cppclass cYlmCache "YlmCache":
        double* data
        int size
        int l_max
        cSpGrid grid
        cYlmCache(cSpGrid grid, int l_max);
        double operator()(int l, int m, int ic)
        double operator()(int l, int m, double c)

    int pow_minus_one(int p)
    double clebsch_gordan_coef(int j1, int m1, int j2, int m2, int J, int M)
    double y3(int l1, int m1, int l2, int m2, int L, int M)

    void sh_series[T](ArraySp2D[T]* arr, int l, int m, T* series, cYlmCache* ylm_cache);

cdef class YlmCache:
    cdef cYlmCache* cdata

cdef class JlCache:
    cdef cJlCache* cdata
