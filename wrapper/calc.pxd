from workspace cimport WfBase, orbs
from wavefunc cimport cShWavefunc
from orbitals cimport cOrbitals
from field cimport field_t
from types cimport sh_f
from atom cimport cAtomCache


cdef extern from "calc.h":
    double calc_wf_ionization_prob(cShWavefunc* wf)
    double calc_orbs_ionization_prob(cOrbitals* orbs)

    double calc_wf_az(cShWavefunc* wf, cAtomCache& atom, field_t* field, double t)
    double calc_wf_az_with_polarization(
            cShWavefunc* wf,
            cAtomCache& atom_cache,
            double* Upol,
            double* dUpol_dr,
            field_t* field,
            double t
    )

    double calc_orbs_az(cOrbitals* orbs, cAtomCache& atom, field_t* field, double t)
    void calc_orbs_az_ne(cOrbitals* orbs, cAtomCache& atom, field_t* field, double t, double* az)

    double calc_wf_jrcd(
            WfBase* ws,
            cShWavefunc* wf,
            cAtomCache& atom,
            field_t* field,
            int Nt,
            double dt,
            double t_smooth
    )

    double calc_orbs_jrcd(
            orbs* ws,
            cOrbitals* orbs,
            cAtomCache& atom,
            field_t* field,
            int Nt,
            double dt,
            double t_smooth
    )

cdef extern from "utils.h":
    double smoothstep(double x, double x0, double x1)
    void selectGpuDevice(int id)
