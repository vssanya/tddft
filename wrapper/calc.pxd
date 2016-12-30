from workspace cimport sphere_kn_workspace_t
from wavefunc cimport sphere_wavefunc_t
from field cimport field_t
from types cimport sphere_pot_t

cdef extern from "calc.h":
    void calc_a(int Nt, double* a, sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t field)

cdef extern from "jrcd.h":
    double jrcd(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t E, sphere_pot_t dUdz, int Nt)
