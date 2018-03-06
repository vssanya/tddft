from types cimport cdouble, sh_f

from grid cimport cShGrid, cSpGrid
from abs_pot cimport uabs_sh_t, Uabs
from wavefunc cimport ShWavefunc, cCtWavefunc, cShWavefunc

from field cimport field_t
from orbitals cimport cOrbitals
from sphere_harmonics cimport cYlmCache
from atom cimport cAtom, cAtomCache, Atom, AtomCache
from hartree_potential cimport potential_xc_f


cdef extern from "eigen.h":
    ctypedef struct eigen_ws_t:
        cShGrid* grid
        double* evec
        double* eval

    eigen_ws_t* eigen_ws_alloc(cShGrid* grid)
    void eigen_ws_free(eigen_ws_t* ws)
    void eigen_calc(eigen_ws_t* ws, sh_f u, int Z)
    void eigen_calc_for_atom(eigen_ws_t* ws, cAtomCache* atom)
    int eigen_get_n_with_energy(eigen_ws_t* ws, double energy)

cdef extern from "workspace/sfa.h" namespace "workspace::sfa":
    cdef cppclass momentum_space:
        momentum_space()
        void propagate(cCtWavefunc& wf, field_t* field, double t, double dt)

cdef extern from "workspace.h":
    ctypedef struct ws_gps_t:
        cShGrid* grid;
        cAtom* atom;

        double dt;
        double e_max;

        cdouble* s;
        int n_evec;

        cShWavefunc* prop_wf;

    ws_gps_t* ws_gps_alloc(cShGrid* grid, cAtom* atom, double dt, double e_max);
    void ws_gps_free(ws_gps_t* ws);
    void ws_gps_calc_s(ws_gps_t* ws, eigen_ws_t* eigen);
    void ws_gps_prop(ws_gps_t* ws, cShWavefunc* wf);
    void ws_gps_prop_common(ws_gps_t* ws, cShWavefunc* wf, uabs_sh_t* uabs, field_t* field, double t);

cdef extern from "workspace.h" namespace "workspace":
    cdef cppclass WfBase:
        WfBase(cAtomCache* atom, cShGrid* grid, uabs_sh_t* uabs, int num_threads)

        void prop_ang(cShWavefunc& wf, double dt, int l, double E)
        #void prop_at(cShWavefunc& wf, cdouble dt, sh_f Ul, int Z, potential_type_e u_type)
        void prop_mix(cShWavefunc& wf, sh_f Al, double dt, int l)
        void prop_abs(cShWavefunc& wf, double dt)
        #void prop_common(cShWavefunc& wf, cdouble dt, int l_max, sh_f* Ul, int Z, potential_type_e u_type, sh_f* Al)
        void prop(cShWavefunc& wf, field_t* field, double t, double dt)
        void prop_img(cShWavefunc& wf, double dt)

        cShGrid* grid
        uabs_sh_t* uabs
        cdouble* alpha
        cdouble* betta
        int num_threads
        cAtomCache atom_cache

    cdef cppclass WfEWithSource:
        WfEWithSource(cAtomCache* atom_cache, cShGrid* grid, uabs_sh_t* uabs, cShWavefunc& source, double E, int num_threads)
        void prop(cShWavefunc& wf, field_t* field, double t, double dt)

    cdef cppclass WfA:
        WfA(cAtomCache* atom, cShGrid* grid, uabs_sh_t* uabs, int num_threads)
        void prop(cShWavefunc& wf, field_t* field, double t, double dt)
        void prop_img(cShWavefunc& wf, double dt)

    cdef cppclass orbs:
        orbs(cAtomCache* atom, cShGrid* sh_grid, cSpGrid* sp_grid, uabs_sh_t* uabs, cYlmCache* ylm_cache, int Uh_lmax, int Uxc_lmax, potential_xc_f Uxc, int num_threads)

        void prop(cOrbitals* orbs, field_t* field, double t, double dt, bint calc_uee)
        void prop_img(cOrbitals* orbs, double dt)
        void calc_Uee(cOrbitals* orbs, int Uxc_lmax, int Uh_lmax)

        WfBase* wf_ws
        double* Uh
        double* Uxc
        cShGrid* sh_grid
        cSpGrid* sp_grid
        double* uh_tmp
        double* n_sp
        double* Uee
        cYlmCache* ylm_cache
        int Uh_lmax
        int Uxc_lmax


cdef class SKnWorkspace:
    cdef:
        WfBase* cdata
        Uabs uabs
        AtomCache atom_cache

cdef class SKnAWorkspace:
    cdef:
        WfA* cdata
        Uabs uabs
        AtomCache atom_cache

cdef class SKnWithSourceWorkspace:
    cdef:
        WfEWithSource* cdata
        ShWavefunc source
        Uabs uabs
        AtomCache atom_cache

cdef class SOrbsWorkspace:
    cdef:
        orbs* cdata
        Uabs uabs
        AtomCache atom_cache

cdef class GPSWorkspace:
    cdef:
        ws_gps_t* cdata

cdef class Eigen:
    cdef:
        eigen_ws_t* cdata

cdef class SFAWorkspace:
    cdef:
        momentum_space* cdata
