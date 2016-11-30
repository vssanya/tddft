from wavefunc cimport sphere_wavefunc_t

cdef extern from "hydrogen.h":
    double hydrogen_U(double r)
    double hydrogen_dUdz(double r)
    void hydrogen_ground(sphere_wavefunc_t* wf)
