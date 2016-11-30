from workspace cimport sphere_kn_workspace_t
from wavefunc cimport sphere_wavefunc_t
from field cimport field_t

cdef extern from "calc.h":
    void calc_a(int Nt, double* a, sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t field)
