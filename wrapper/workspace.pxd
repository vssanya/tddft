from types cimport cdouble, sh_f

from grid cimport sh_grid_t, sp_grid_t
from abs_pot cimport uabs_sh_t, Uabs
from wavefunc cimport sh_wavefunc_t

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

    ctypedef struct ws_wf_t:
        sh_grid_t* grid
        uabs_sh_t* uabs
        cdouble* alpha
        cdouble* betta
        int num_threads

    ctypedef struct ws_orbs_t:
        ws_wf_t* wf_ws
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

    ws_wf_t* ws_wf_new(
            sh_grid_t* sh_grid,
            uabs_sh_t* uabs,
            int num_threads
            )

    void ws_wf_del(ws_wf_t* ws)

    void ws_wf_prop_ang(
            ws_wf_t* ws,
            sh_wavefunc_t* wf,
            double dt,
            int l, double E)

    void ws_wf_prop_at(
            ws_wf_t* ws,
            sh_wavefunc_t* wf,
            cdouble dt,
            sh_f Ul,
            uabs_sh_t* uabs
            )

    void ws_wf_prop_at_v2(
            ws_wf_t* ws,
            sh_wavefunc_t* wf,
            cdouble dt,
            sh_f Ul,
            uabs_sh_t* uabs
            )

    void ws_wf_prop(
            ws_wf_t* ws,
            sh_wavefunc_t* wf,
            atom_t* atom,
            field_t* E,
            double t,
            double dt
            )

    void ws_wf_prop_img(
            ws_wf_t* ws,
            sh_wavefunc_t* wf,
            atom_t* atom,
            double dt
            )

    void ws_orbs_prop(
            ws_orbs_t* ws,
            orbitals_t* orbs,
            atom_t* atom,
            field_t* field,
            double t,
            double dt,
            bint calc_uee
            )

    void ws_orbs_prop_img(
        ws_orbs_t* ws,
        orbitals_t* orbs,
        atom_t* atom,
        double dt
    )

    ws_orbs_t* ws_orbs_alloc(
        sh_grid_t* sh_grid,
        sp_grid_t* sp_grid,
        uabs_sh_t* uabs,
        ylm_cache_t* ylm_cache,
        int Uh_lmax,
        int Uxc_lmax,
        potential_xc_f uxc,
        int num_threads
    )

    void ws_orbs_free(ws_orbs_t* ws)

    void ws_orbs_calc_Uee(ws_orbs_t* ws, orbitals_t* orbs, int Uxc_lmax, int Uh_lmax)


cdef class SKnWorkspace:
    cdef:
        ws_wf_t* cdata
        Uabs uabs

cdef class SOrbsWorkspace:
    cdef:
        ws_orbs_t* cdata
        Uabs uabs

cdef class GPSWorkspace:
    cdef:
        ws_gps_t* cdata

cdef class Eigen:
    cdef:
        eigen_ws_t* cdata
