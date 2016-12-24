from types cimport sphere_pot_t, cdouble
from grid cimport sh_grid_t

cdef extern from "sphere_wavefunc.h":
    ctypedef struct sphere_wavefunc_t:
        sh_grid_t* grid
        cdouble* data
        int m

    sphere_wavefunc_t* sphere_wavefunc_new(sh_grid_t* grid, int m)
    sphere_wavefunc_t* sphere_wavefunc_new_from(cdouble* data, sh_grid_t* grid, int m)

    void   sphere_wavefunc_del(sphere_wavefunc_t* wf)
    double sphere_wavefunc_norm(sphere_wavefunc_t* wf)
    void   sphere_wavefunc_normalize(sphere_wavefunc_t* wf)
    void   sphere_wavefunc_print(sphere_wavefunc_t * wf)
    double sphere_wavefunc_cos(
		sphere_wavefunc_t * wf,
		sphere_pot_t U
    )
    double sphere_wavefunc_z(sphere_wavefunc_t * wf)
    cdouble swf_get_sp(sphere_wavefunc_t* wf, int i[3]);

cdef class SWavefunc:
    cdef sphere_wavefunc_t* data
