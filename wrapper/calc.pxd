from workspace cimport sh_workspace_t, sh_orbs_workspace_t
from wavefunc cimport sh_wavefunc_t
from orbitals cimport orbitals_t
from field cimport field_t
from types cimport sh_f
from atom cimport atom_t

cdef extern from "calc.h":
    double calc_wf_ionization_prob(sh_wavefunc_t* wf)
    double calc_orbs_ionization_prob(orbitals_t* orbs)
    double calc_wf_az(sh_wavefunc_t* wf, atom_t* atom, field_t field, double t)
    double calc_orbs_az(orbitals_t* orbs, atom_t* atom, field_t field, double t)
    void calc_orbs_az_ne(orbitals_t* orbs, field_t field, double t, double* az)
    void calc_wf_az_t(
            int Nt, double* a,
            sh_workspace_t* ws,
            sh_wavefunc_t* wf,
            field_t field,
            double dt);

    double calc_wf_jrcd(
            sh_workspace_t* ws,
            sh_wavefunc_t* wf,
            atom_t* atom,
            field_t field,
            int Nt, 
            double dt,
            double t_smooth
    )

    double calc_orbs_jrcd(
            sh_orbs_workspace_t* ws,
            orbitals_t* orbs,
            atom_t* atom,
            field_t field,
            int Nt, 
            double dt,
            double t_smooth
    )

cdef extern from "utils.h":
    double smoothstep(double x, double x0, double x1)
