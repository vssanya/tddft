from libcpp cimport bool as bool_t

from types cimport cdouble, sh_f, optional, none

from grid cimport cShGrid, cSpGrid, cShNeGrid, cRange, cGrid3d
from abs_pot cimport cUabsCache, UabsCache, UabsNeCache
from wavefunc cimport ShWavefunc, cCtWavefunc, cShWavefunc, Wavefunc, WavefuncArray, cCartWavefunc3D, CartWavefunc3D

from field cimport field_t
from orbitals cimport Orbitals
from sphere_harmonics cimport cYlmCache
from atom cimport cAtom, Atom, ShAtomCache, ShNeAtomCache, AtomCache
from carray cimport Array2D, Array3D
from hartree_potential cimport XCPotentialEnum


cdef extern from "eigen.h":
    ctypedef struct eigen_ws_t:
        cShGrid* grid
        double* evec
        double* eval

    eigen_ws_t* eigen_ws_alloc(cShGrid* grid)
    void eigen_ws_free(eigen_ws_t* ws)
    void eigen_calc(eigen_ws_t* ws, sh_f u, int Z)
    void eigen_calc_for_atom(eigen_ws_t* ws, AtomCache[cShGrid]* atom)
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
    void ws_gps_prop_common(ws_gps_t* ws, cShWavefunc* wf, cUabsCache* uabs, field_t* field, double t);

cdef extern from "workspace.h" namespace "workspace":
    cdef cppclass PropAtType:
        pass

    cdef cppclass Gauge:
        pass

    cdef cppclass CartWavefuncWS:
        CartWavefuncWS(cCartWavefunc3D* wf, cAtom& atom, Array3D[double]* uabs);

        void prop_abs(double dt);
        void prop(field_t* field, double t, double dt);
        void prop_img(double dt);
        void prop_r(double E[3], cdouble dt);
        void prop_r_norm(double E[3], double norm, cdouble dt);
        void prop_r_norm_abs(double E[3], double norm, double dt);

        cCartWavefunc3D* wf;
        cGrid3d& grid;

        cAtom& atom;
        Array3D[double]& uabs;

    cdef cppclass WavefuncWS[Grid]:
        WavefuncWS(
            Grid      & grid,
            AtomCache[Grid]* atom_cache,
            cUabsCache& uabs,
            PropAtType propAtType,
            Gauge gauge,
            int num_threads
        )

        void set_atom_cache(AtomCache[Grid]* atom_cache)

        void prop_ang(Wavefunc[Grid]& wf, double dt, int l, double E)
        #void prop_at(cShWavefunc& wf, cdouble dt, sh_f Ul, int Z, potential_type_e u_type)
        void prop_mix(Wavefunc[Grid]& wf, sh_f Al, double dt, int l)
        void prop_abs(Wavefunc[Grid]& wf, double dt)
        #void prop_common(cShWavefunc& wf, cdouble dt, int l_max, sh_f* Ul, int Z, potential_type_e u_type, sh_f* Al)
        void prop(Wavefunc[Grid]& wf, field_t* field, double t, double dt)
        void prop_without_field(Wavefunc[Grid]& wf, double dt)
        void prop_img(Wavefunc[Grid]& wf, double dt)

        Grid* grid
        cUabsCache* uabs
        cdouble* alpha
        cdouble* betta
        int num_threads
        AtomCache[Grid]* atom_cache

    cdef cppclass WfEWithSource:
        WfEWithSource(
            cShGrid    & grid,
            AtomCache[cShGrid]* atom_cache,
            cUabsCache & uabs,
            cShWavefunc& wf_source,
            double E,
            PropAtType propAtType,
            Gauge gauge,
            int num_threads
        )
        void prop(cShWavefunc& wf, field_t* field, double t, double dt)

        double abs_norm

    cdef cppclass WfWithPolarization:
        WfWithPolarization(
            cShGrid   & grid,
            AtomCache[cShGrid]* atom_cache,
            cUabsCache& uabs,
            double* Upol_1,
            double* Upol_2,
            PropAtType propAtType,
            Gauge gauge,
            int num_threads
        )
        void prop(cShWavefunc& wf, field_t* field, double t, double dt)
        void prop_without_field(cShWavefunc& wf, double dt)
        void prop_img(cShWavefunc& wf, double dt)

    cdef cppclass OrbitalsWS[Grid]:
        OrbitalsWS(
            Grid      & sh_grid,
            cSpGrid   & sp_grid,
            AtomCache[Grid]* atom_cache,
            cUabsCache& uabs,
            cYlmCache & ylm_cache,
            int Uh_lmax,
            int Uxc_lmax,
            XCPotentialEnum potentialType,
            PropAtType propAtType,
            Gauge gauge,
            int num_threads
        )

        void setTimeApproxUeeTwoPointFor(Orbitals[Grid]& orbs)
        void prop(Orbitals[Grid]& orbs, field_t* field, double t, double dt, bint calc_uee, bool_t* activeOrbs, int* dt_count)
        void prop_img(Orbitals[Grid]& orbs, double dt, bool_t* activeOrbs, int* dt_count, bint calc_uee)
        void prop_ha(Orbitals[Grid]& orbs, double dt, bint calc_uee, bool_t* activeOrbs)
        void prop_abs(Orbitals[Grid]& orbs, double dt, bool_t* activeOrbs)
        void calc_Uee(Orbitals[Grid]& orbs, int Uxc_lmax, int Uh_lmax, Array2D[double]* Uee, optional[cRange] rRange)

        WavefuncWS[Grid]* wf_ws
        double* Uh
        double* Uxc
        Grid& sh_grid
        cSpGrid& sp_grid
        double* uh_tmp
        double* n_sp
        Array2D[double]* Uee
        cYlmCache& ylm_cache
        int Uh_lmax
        int Uxc_lmax


cdef extern from "workspace.h" namespace "workspace::PropAtType":
    cdef PropAtType Odr3
    cdef PropAtType Odr4

cdef extern from "workspace.h" namespace "workspace::Gauge":
    cdef Gauge LENGTH
    cdef Gauge VELOCITY

cdef class CartWorkspace:
    cdef:
        CartWavefuncWS* cdata

cdef class ShWavefuncWS:
    cdef:
        WavefuncWS[cShGrid]* cdata
        UabsCache uabs
        ShAtomCache atom_cache

cdef class ShNeWavefuncWS:
    cdef:
        WavefuncWS[cShNeGrid]* cdata
        UabsNeCache uabs
        ShNeAtomCache atom_cache

cdef class WfWithPolarizationWorkspace:
    cdef:
        WfWithPolarization* cdata
        UabsCache uabs
        ShAtomCache atom_cache
        double[:] Upol_1
        double[:] Upol_2

cdef class SKnWithSourceWorkspace:
    cdef:
        WfEWithSource* cdata
        ShWavefunc source
        UabsCache uabs
        ShAtomCache atom_cache

cdef class ShOrbitalsWS:
    cdef:
        OrbitalsWS[cShGrid]* cdata
        UabsCache uabs
        ShAtomCache atom_cache

cdef class ShNeOrbitalsWS:
    cdef:
        OrbitalsWS[cShNeGrid]* cdata
        UabsNeCache uabs
        ShNeAtomCache atom_cache

cdef class GPSWorkspace:
    cdef:
        ws_gps_t* cdata

cdef class Eigen:
    cdef:
        eigen_ws_t* cdata

cdef class SFAWorkspace:
    cdef:
        momentum_space* cdata

cdef extern from "workspace/wf_array.h" namespace "workspace":
    cdef cppclass WfArray[Grid]:
        WfArray(
            Grid      & grid,
            AtomCache[Grid]* atom_cache,
            cUabsCache& uabs,
            PropAtType propAtType,
            Gauge gauge,
            int num_threads
        )
        void prop(WavefuncArray[Grid]* wf, double* E, double dt)


cdef class ShWfArrayWS:
    cdef:
        WfArray[cShGrid]* cdata

        UabsCache uabs
        ShAtomCache atom_cache

cdef class ShNeWfArrayWS:
    cdef:
        WfArray[cShNeGrid]* cdata

        UabsNeCache uabs
        ShNeAtomCache atom_cache
