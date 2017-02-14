from workspace cimport sphere_kn_workspace_t
from wavefunc cimport sphere_wavefunc_t
from field cimport field_t
from types cimport sphere_pot_t

cdef extern from "calc.h":
    double calc_az(sphere_wavefunc_t* wf, field_t field, sphere_pot_t dudz, double t)
    double calc_az_lf(sphere_wavefunc_t* wf, field_t field, sphere_pot_t dudz, double t)
    void calc_az_t(
            int Nt, double* a,
            sphere_kn_workspace_t* ws,
            sphere_wavefunc_t* wf,
            field_t field);

cdef extern from "jrcd.h":
    double jrcd(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t E, sphere_pot_t dUdz, int Nt)

cdef extern from "utils.h":
    double smoothstep(double x, double x0, double x1)
