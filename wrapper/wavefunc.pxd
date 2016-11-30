from types cimport sphere_pot_t
from grid cimport sphere_grid_t

cdef extern from "sphere_wavefunc.h":
    ctypedef struct sphere_wavefunc_t:
        pass

    sphere_wavefunc_t* sphere_wavefunc_alloc(
		sphere_grid_t* grid,
		int m
    )
    void   sphere_wavefunc_free(sphere_wavefunc_t* wf)
    double sphere_wavefunc_norm(sphere_wavefunc_t* wf)
    void   sphere_wavefunc_normalize(sphere_wavefunc_t* wf)
    void   sphere_wavefunc_print(sphere_wavefunc_t * wf)
    double sphere_wavefunc_cos(
		sphere_wavefunc_t * wf,
		sphere_pot_t U
    )
    double sphere_wavefunc_z(sphere_wavefunc_t * wf)

cdef class SWavefunc:
    cdef sphere_wavefunc_t* data
