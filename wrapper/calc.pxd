from workspace cimport sh_workspace_t
from wavefunc cimport sphere_wavefunc_t
from orbitals cimport ks_orbitals_t
from field cimport field_t
from types cimport sh_f

cdef extern from "calc.h":
    double calc_ionization_prob(ks_orbitals_t* orbs)
    double calc_az(sphere_wavefunc_t* wf, field_t field, sh_f dudz, double t)
    void calc_az_t(
            int Nt, double* a,
            sh_workspace_t* ws,
            sphere_wavefunc_t* wf,
            field_t field,
            double dt);

cdef extern from "jrcd.h":
    double jrcd(
            sh_workspace_t* ws,
            sphere_wavefunc_t* wf,
            field_t E,
            sh_f dudz, int Nt,
            double dt,
            double t_smooth)

cdef extern from "utils.h":
    double smoothstep(double x, double x0, double x1)
