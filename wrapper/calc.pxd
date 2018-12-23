from workspace cimport WavefuncWS, OrbitalsWS
from wavefunc cimport Wavefunc, cShWavefunc
from orbitals cimport Orbitals
from field cimport field_t
from types cimport sh_f
from atom cimport cAtomCache


cdef extern from "calc.h":
    double calc_wf_az[Grid](Wavefunc[Grid]* wf, cAtomCache& atom, field_t* field, double t)
    double calc_wf_az_with_polarization(
            cShWavefunc* wf,
            cAtomCache& atom_cache,
            double* Upol,
            double* dUpol_dr,
            field_t* field,
            double t
    )

    double calc_orbs_az[Grid](Orbitals[Grid]& orbs, cAtomCache& atom, field_t* field, double t)
    void calc_orbs_az_ne[Grid](Orbitals[Grid]* orbs, cAtomCache& atom, field_t* field, double t, double* az)

cdef extern from "utils.h":
    double smoothstep(double x, double x0, double x1)
    void selectGpuDevice(int id)
