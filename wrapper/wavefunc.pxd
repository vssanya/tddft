from types cimport sh_f, cdouble
from grid cimport cShGrid, cShNeGrid, cSpGrid, cGrid2d
from grid cimport ShGrid, ShNeGrid, SpGrid2d
from sphere_harmonics cimport cYlmCache


cdef extern from "wavefunc/cartesian_2d.h":
    cdef cppclass cCtWavefunc "CtWavefunc":
        cGrid2d* grid,
        double* data,
        bint data_own

        cCtWavefunc()
        cCtWavefunc(cGrid2d* grid)

        double norm()


cdef extern from "sh_wavefunc.h":
    cdef cppclass Wavefunc[Grid]:
        Grid* grid

        cdouble* data
        bint data_own

        int m

        Wavefunc(cdouble* data, Grid* grid, int m)
        Wavefunc(Grid* grid, int m)

        double abs_2(int ir, int il)
        void copy(Wavefunc* wf_dest)
        cdouble operator*(Wavefunc& other)
        void exclude(Wavefunc& other)
        double cos(sh_f func)
        void   cos_r(sh_f U, double* res)
        double cos_r2(sh_f U, int Z)

        double norm(sh_f mask)
        double norm()

        void normalize()

        double z(sh_f mask)
        double z()

        cdouble pz()
        void random_l(int l)
        cdouble get_sp(cSpGrid* grid, int i[3], cYlmCache* ylm_cache)
        void n_sp(cSpGrid* grid, double* n, cYlmCache* ylm_cache)
        @staticmethod
        void ort_l(int l, int n, Wavefunc** wfs)

    ctypedef Wavefunc[cShGrid] cShWavefunc "ShWavefunc"
    ctypedef Wavefunc[cShNeGrid] cShNeWavefunc "ShNeWavefunc"

cdef class ShWavefunc:
    cdef cShWavefunc* cdata
    cdef bint dealloc
    cdef public ShGrid grid

    cdef _set_data(self, cShWavefunc* data)

cdef class ShNeWavefunc:
    cdef cShNeWavefunc* cdata
    cdef bint dealloc
    cdef public ShNeGrid grid

    cdef _set_data(self, cShNeWavefunc* data)

cdef class CtWavefunc:
    cdef cCtWavefunc* cdata
    cdef public SpGrid2d grid

cdef ShWavefunc swavefunc_from_point(cShWavefunc* data, ShGrid grid, bint dealloc)
