from types cimport cdouble, sh_f

from grid cimport sh_grid_t, sp_grid_t
from abs_pot cimport uabs_sh_t, Uabs
from wavefunc cimport sh_wavefunc_t, ct_wavefunc_t

from field cimport field_t
from orbitals cimport orbitals_t
from sphere_harmonics cimport ylm_cache_t
from atom cimport atom_t
from hartree_potential cimport potential_xc_f


cdef extern from "eigen.h":
    ctypedef struct eigen_ws_t:
        sh_grid_t* grid
        double* evec
        double* eval

    eigen_ws_t* eigen_ws_alloc(sh_grid_t* grid)
    void eigen_ws_free(eigen_ws_t* ws)
    void eigen_calc(eigen_ws_t* ws, sh_f u, int Z)
    void eigen_calc_for_atom(eigen_ws_t* ws, atom_t* atom)
    int eigen_get_n_with_energy(eigen_ws_t* ws, double energy);

cdef extern from "workspace/sfa.h" namespace "workspace::sfa":
    cdef cppclass momentum_space:
        momentum_space()
        void propagate(ct_wavefunc_t& wf, field_t* field, double t, double dt)

cdef extern from "workspace.h":
    ctypedef struct ws_gps_t:
        sh_grid_t* grid;
        atom_t* atom;

        double dt;
        double e_max;

        cdouble* s;
        int n_evec;

        sh_wavefunc_t* prop_wf;

    ws_gps_t* ws_gps_alloc(sh_grid_t* grid, atom_t* atom, double dt, double e_max);
    void ws_gps_free(ws_gps_t* ws);
    void ws_gps_calc_s(ws_gps_t* ws, eigen_ws_t* eigen);
    void ws_gps_prop(ws_gps_t* ws, sh_wavefunc_t* wf);
    void ws_gps_prop_common(ws_gps_t* ws, sh_wavefunc_t* wf, uabs_sh_t* uabs, field_t* field, double t);

cdef extern from "workspace.h" namespace "workspace":
    cdef cppclass wf_base:
        wf_base()
        wf_base(sh_grid_t* grid, uabs_sh_t* uabs, int num_threads)

        void prop_ang(sh_wavefunc_t& wf, double dt, int l, double E)
        #void prop_at(sh_wavefunc_t& wf, cdouble dt, sh_f Ul, int Z, potential_type_e u_type)
        void prop_mix(sh_wavefunc_t& wf, sh_f Al, double dt, int l)
        void prop_abs(sh_wavefunc_t& wf, double dt)
        #void prop_common(sh_wavefunc_t& wf, cdouble dt, int l_max, sh_f* Ul, int Z, potential_type_e u_type, sh_f* Al)
        void prop(sh_wavefunc_t& wf, atom_t* atom, field_t* field, double t, double dt)
        void prop_img(sh_wavefunc_t& wf, atom_t* atom, double dt)

        sh_grid_t* grid
        uabs_sh_t* uabs
        cdouble* alpha
        cdouble* betta
        int num_threads

    cdef cppclass wf_A:
        wf_A()
        wf_A(sh_grid_t* grid, uabs_sh_t* uabs, int num_threads)
        void prop(sh_wavefunc_t& wf, atom_t* atom, field_t* field, double t, double dt)
        void prop_img(sh_wavefunc_t& wf, atom_t* atom, double dt)

    cdef cppclass orbs:
        orbs()
        orbs(sh_grid_t* sh_grid, sp_grid_t* sp_grid, uabs_sh_t* uabs, ylm_cache_t* ylm_cache, int Uh_lmax, int Uxc_lmax, potential_xc_f Uxc, int num_threads)

        void prop(orbitals_t* orbs, atom_t* atom, field_t* field, double t, double dt, bint calc_uee)
        void prop_img(orbitals_t* orbs, atom_t* atom, double dt)
        void calc_Uee(orbitals_t* orbs, int Uxc_lmax, int Uh_lmax)

        wf_base* wf_ws
        double* Uh
        double* Uxc
        sh_grid_t* sh_grid
        sp_grid_t* sp_grid
        double* uh_tmp
        double* n_sp
        double* Uee
        ylm_cache_t* ylm_cache
        int Uh_lmax
        int Uxc_lmax


cdef class SKnWorkspace:
    cdef:
        wf_base* cdata
        Uabs uabs

cdef class SKnAWorkspace:
    cdef:
        wf_A* cdata
        Uabs uabs

cdef class SOrbsWorkspace:
    cdef:
        orbs* cdata
        Uabs uabs

cdef class GPSWorkspace:
    cdef:
        ws_gps_t* cdata

cdef class Eigen:
    cdef:
        eigen_ws_t* cdata

cdef class SFAWorkspace:
    cdef:
        momentum_space* cdata
