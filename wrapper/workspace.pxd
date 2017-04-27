from types cimport cdouble, sh_f

from grid cimport sh_grid_t, sp_grid_t
from abs_pot cimport uabs_sh_t, Uabs
from wavefunc cimport sh_wavefunc_t

from field cimport field_t
from orbitals cimport orbitals_t
from sphere_harmonics cimport ylm_cache_t
from atom cimport atom_t

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

cdef extern from "sh_workspace.h":
    ctypedef struct gps_ws_t:
        sh_grid_t* grid;
        atom_t* atom;

        double dt;
        double e_max;

        cdouble* s;
        int n_evec;

        sh_wavefunc_t* prop_wf;

    gps_ws_t* gps_ws_alloc(sh_grid_t* grid, atom_t* atom, double dt, double e_max);
    void gps_ws_free(gps_ws_t* ws);
    void gps_ws_calc_s(gps_ws_t* ws, eigen_ws_t* eigen);
    void gps_ws_prop(gps_ws_t* ws, sh_wavefunc_t* wf);
    void gps_ws_prop_common(gps_ws_t* ws, sh_wavefunc_t* wf, uabs_sh_t* uabs, field_t* field, double t);

    ctypedef struct sh_workspace_t:
        sh_grid_t* grid
        uabs_sh_t* uabs
        cdouble* alpha
        cdouble* betta
        int num_threads

    ctypedef struct sh_orbs_workspace_t:
        sh_workspace_t* wf_ws
        double* Uh
        double* Uxc
        sh_grid_t* sh_grid;
        sp_grid_t* sp_grid;
        double* uh_tmp
        double* n_sp
        ylm_cache_t* ylm_cache;

    sh_workspace_t* sh_workspace_alloc(
            sh_grid_t* sh_grid,
            uabs_sh_t* uabs,
            int num_threads
            )

    void sh_workspace_free(sh_workspace_t* ws)

    void sh_workspace_prop_ang(
            sh_workspace_t* ws,
            sh_wavefunc_t* wf,
            double dt,
            int l, double E)

    void sh_workspace_prop_at(
            sh_workspace_t* ws,
            sh_wavefunc_t* wf,
            cdouble dt,
            sh_f Ul,
            uabs_sh_t* uabs
            )

    void sh_workspace_prop_at_v2(
            sh_workspace_t* ws,
            sh_wavefunc_t* wf,
            cdouble dt,
            sh_f Ul,
            uabs_sh_t* uabs
            )

    void sh_workspace_prop(
            sh_workspace_t* ws,
            sh_wavefunc_t* wf,
            atom_t* atom,
            field_t* E,
            double t,
            double dt
            )

    void sh_workspace_prop_img(
            sh_workspace_t* ws,
            sh_wavefunc_t* wf,
            atom_t* atom,
            double dt
            )

    void sh_orbs_workspace_prop(
            sh_orbs_workspace_t* ws,
            orbitals_t* orbs,
            atom_t* atom,
            field_t* field,
            double t,
            double dt
            )
    void sh_orbs_workspace_prop_img(
            sh_orbs_workspace_t* ws,
            orbitals_t* orbs,
            atom_t* atom,
            double dt
            )

    sh_orbs_workspace_t* sh_orbs_workspace_alloc(
            sh_grid_t* sh_grid,
            sp_grid_t* sp_grid,
            uabs_sh_t* uabs,
            ylm_cache_t* ylm_cache,
            int num_threads
            )
    void sh_orbs_workspace_free(sh_orbs_workspace_t* ws)


cdef class SKnWorkspace:
    cdef:
        sh_workspace_t* cdata
        Uabs uabs

cdef class SOrbsWorkspace:
    cdef:
        sh_orbs_workspace_t* cdata
        Uabs uabs

cdef class GPSWorkspace:
    cdef:
        gps_ws_t* cdata

cdef class Eigen:
    cdef:
        eigen_ws_t* cdata
