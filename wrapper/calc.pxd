from workspace cimport WavefuncWS, OrbitalsWS
from wavefunc cimport Wavefunc, cShWavefunc
from orbitals cimport Orbitals
from field cimport field_t
from types cimport sh_f, cdouble
from atom cimport AtomCache
from carray cimport Array2D


cdef extern from "calc.h":
    double calc_wf_az[Grid](Wavefunc[Grid]* wf, AtomCache[Grid]& atom, field_t* field, double t)
    double calc_wf_az[Grid](Wavefunc[Grid]* wf_p, Wavefunc[Grid]* wf_g, AtomCache[Grid]& atom, int l_max)
    double calc_wf_az_with_polarization[Grid](
            Wavefunc[Grid]* wf,
            AtomCache[Grid]& atom_cache,
            double* Upol,
            double* dUpol_dr,
            field_t* field,
            double t
    )

    void calc_orbs_az_ne_Vee_0[Grid](Orbitals[Grid]* orbs, Array2D[double]& Uee, Array2D[double]& dUeedr, AtomCache& atom_cache, field_t* field, double t, double* az);
    void calc_orbs_az_ne_Vee_1[Grid](Orbitals[Grid]* orbs, Array2D[double]& Uee, Array2D[double]& dUeedr, AtomCache& atom_cache, field_t* field, double t, double* az);

    double calc_orbs_az[Grid](Orbitals[Grid]& orbs, AtomCache[Grid]& atom, field_t* field, double t)
    void calc_orbs_az_ne[Grid](Orbitals[Grid]* orbs, AtomCache[Grid]& atom, field_t* field, double t, double* az)

cdef extern from "utils.h":
    double smoothstep(double x, double x0, double x1)
    void selectGpuDevice(int id)
