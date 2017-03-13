from types cimport sh_f, cdouble
from grid cimport sh_grid_t, sp_grid_t
from grid cimport ShGrid
from sphere_harmonics cimport ylm_cache_t


cdef extern from "sh_wavefunc.h":
    ctypedef struct sh_wavefunc_t:
        sh_grid_t* grid

        cdouble* data
        bint data_own

        int m

    sh_wavefunc_t* sh_wavefunc_new(sh_grid_t* grid, int m)
    sh_wavefunc_t* sh_wavefunc_new_from(cdouble* data, sh_grid_t* grid, int m)

    void   sh_wavefunc_del(sh_wavefunc_t* wf)
    void   sh_wavefunc_n_sp(sh_wavefunc_t* wf, sp_grid_t* grid, double* n, ylm_cache_t* ylm_cache)
    double sh_wavefunc_norm(sh_wavefunc_t* wf)
    void   sh_wavefunc_normalize(sh_wavefunc_t* wf)
    void   sh_wavefunc_print(sh_wavefunc_t * wf)
    double sh_wavefunc_cos(
		sh_wavefunc_t * wf,
		sh_f U
    )
    double sh_wavefunc_z(sh_wavefunc_t * wf)
    cdouble swf_get_sp(sh_wavefunc_t* wf, sp_grid_t* grid, int i[3], ylm_cache_t* ylm_cache)

cdef class SWavefunc:
    cdef sh_wavefunc_t* data
    cdef bint dealloc

    cdef _set_data(self, sh_wavefunc_t* data)

cdef SWavefunc swavefunc_from_point(sh_wavefunc_t* data)
