from types cimport sh_f, cdouble
from grid cimport cShGrid, cShNeGrid, cShNeGrid3D, cSpGrid, cGrid2d, cSpGrid2d
from grid cimport ShGrid, ShNeGrid, SpGrid2d
from sphere_harmonics cimport cYlmCache

from libcpp cimport bool as bool_t

from mpi4py.MPI cimport Comm
from mpi4py.libmpi cimport MPI_Comm


cdef extern from "wavefunc/cartesian_2d.h":
    cdef cppclass cCtWavefunc "CtWavefunc":
        cGrid2d* grid,
        double* data,
        bint data_own

        cCtWavefunc()
        cCtWavefunc(cGrid2d* grid)

        double norm()


cdef extern from "wavefunc/sh_2d.h":
    cdef cppclass Wavefunc[Grid]:
        Grid& grid

        cdouble* data
        bint data_own

        int m

        Wavefunc(cdouble* data, Grid& grid, int m)
        Wavefunc(Grid& grid, int m)

        void init_p(double p, int l_max)

        double abs_2(int ir, int il)
        void copy(Wavefunc* wf_dest)
        cdouble operator*(Wavefunc& other)
        void exclude(Wavefunc& other)
        double cos(sh_f func)
        double cos(sh_f func, Wavefunc& other)
        void   cos_r(sh_f U, double* res)
        double cos_r2(sh_f U, int Z)

        double norm(sh_f mask)
        double norm()
        void norm_z(double* res)

        void normalize()

        double z(sh_f mask)
        double z()

        double z2(sh_f mask)
        double z2()

        cdouble pz()
        void random_l(int l)
        cdouble get_sp(cSpGrid& grid, int i[3], cYlmCache* ylm_cache)
        void n_sp(cSpGrid& grid, double* n, cYlmCache* ylm_cache)
        @staticmethod
        void ort_l(int l, int n, Wavefunc** wfs)

    ctypedef Wavefunc[cShGrid] cShWavefunc "ShWavefunc"
    ctypedef Wavefunc[cShNeGrid] cShNeWavefunc "ShNeWavefunc"
    ctypedef Wavefunc[cSpGrid2d] cSpWavefunc "SpWavefunc2d"

cdef extern from "wavefunc/sh_arr.h":
    cdef cppclass WavefuncArray[Grid]:
        WavefuncArray(int N, Grid& grid, int* m, MPI_Comm mpi_comm, int* const rank)

        void set(cdouble value)
        void set(int index, Wavefunc[Grid]* wf)
        void set_all(Wavefunc[Grid]* wf)

        void norm(double* n)
        void norm(double* n, sh_f mask)
        void normalize(bool_t* activeOrbs, double* norm)

        void z(double* z)
        void z(double* z, sh_f mask)

        void z2(double* z2)
        void z2(double* z2, sh_f mask)

        void cos(double* res, sh_f U)

        int N
        Grid grid
        Wavefunc[Grid]** wf
        cdouble* data
        MPI_Comm mpi_comm
        int mpi_rank
        Wavefunc[Grid]* mpi_wf

cdef class ShWavefuncArray:
    cdef WavefuncArray[cShGrid]* cdata
    cdef Comm mpi_comm
    cdef ShGrid grid

cdef class ShNeWavefuncArray:
    cdef WavefuncArray[cShNeGrid]* cdata
    cdef Comm mpi_comm
    cdef ShNeGrid grid

cdef extern from "sphere_harmonics.h":
    void sp_to_sh(cSpWavefunc* src, cShWavefunc* dest, cYlmCache* ylm_cache, int m)
    void sh_to_sp(cShWavefunc* src, cSpWavefunc* dest, cYlmCache* ylm_cache, int m)

cdef extern from "wavefunc/sh_3d.h":
    cdef cppclass ShWavefunc3D[Grid]:
        Grid& grid

        cdouble* data
        bint data_own

        int m

        ShWavefunc3D(cdouble* data, Grid& grid)
        ShWavefunc3D(Grid& grid)

        double cos(sh_f func)
        double sin_sin(sh_f func)
        double sin_cos(sh_f func)

        double abs_2(int ir, int il)
        cdouble operator*(ShWavefunc3D& other)
        void exclude(ShWavefunc3D& other)

        double norm(sh_f mask)
        double norm()

        void normalize()

    ctypedef ShWavefunc3D[cShNeGrid3D] cShNeWavefunc3D "ShNeWavefunc3D"

cdef class ShWavefunc:
    cdef cShWavefunc* cdata
    cdef bint dealloc
    cdef public ShGrid grid

    cdef _set_data(self, cShWavefunc* data)

cdef class SpWavefunc:
    cdef cSpWavefunc* cdata
    cdef bint dealloc
    cdef public SpGrid2d grid

    cdef _set_data(self, cSpWavefunc* data)

cdef class ShNeWavefunc:
    cdef cShNeWavefunc* cdata
    cdef bint dealloc
    cdef public ShNeGrid grid

    cdef _set_data(self, cShNeWavefunc* data)

cdef class ShNeWavefunc3D:
    cdef cShNeWavefunc3D* cdata
    cdef bint dealloc
    cdef public ShNeGrid grid

    cdef _set_data(self, cShNeWavefunc3D* data)

cdef class CtWavefunc:
    cdef cCtWavefunc* cdata
    cdef public SpGrid2d grid

cdef ShWavefunc sh_wavefunc_from_point(cShWavefunc* data, ShGrid grid, bint dealloc)
cdef ShNeWavefunc shne_wavefunc_from_point(cShNeWavefunc* data, ShNeGrid grid, bint dealloc)
