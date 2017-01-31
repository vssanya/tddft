from wavefunc cimport sphere_wavefunc_t

cdef extern from "hydrogen.h":
    double hydrogen_sh_U(double r)
    double hydrogen_sh_dUdz(double r)
    void hydrogen_ground(sphere_wavefunc_t* wf)
