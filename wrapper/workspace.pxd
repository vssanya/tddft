from types cimport cdouble, sh_f

from grid cimport sh_grid_t, sp_grid_t
from wavefunc cimport sh_wavefunc_t

from field cimport field_t
from orbitals cimport orbitals_t
from sphere_harmonics cimport ylm_cache_t

cdef extern from "sh_workspace.h":
    ctypedef struct sh_workspace_t:
        sh_grid_t* grid
        sh_f U
        sh_f Uabs
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
            sh_f U,
            sh_f Uabs,
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
            sh_f Uabs
            )

    void sh_workspace_prop_at_v2(
            sh_workspace_t* ws,
            sh_wavefunc_t* wf,
            cdouble dt,
            sh_f Ul,
            sh_f Uabs
            )

    void sh_workspace_prop(
            sh_workspace_t* ws,
            sh_wavefunc_t* wf,
            field_t E,
            double t,
            double dt
            )

    void sh_workspace_prop_img(
            sh_workspace_t* ws,
            sh_wavefunc_t* wf,
            double dt
            )

    void sh_orbs_workspace_prop(
            sh_orbs_workspace_t* ws,
            orbitals_t* orbs,
            field_t field,
            double t,
            double dt
            )
    void sh_orbs_workspace_prop_img(
            sh_orbs_workspace_t* ws,
            orbitals_t* orbs,
            double dt
            )

    sh_orbs_workspace_t* sh_orbs_workspace_alloc(
            sh_grid_t* sh_grid,
            sp_grid_t* sp_grid,
            sh_f U,
            sh_f Uabs,
            ylm_cache_t* ylm_cache,
            int num_threads
            )
    void sh_orbs_workspace_free(sh_orbs_workspace_t* ws)


cdef class SKnWorkspace:
    cdef:
        sh_workspace_t* data

cdef class SOrbsWorkspace:
    cdef:
        sh_orbs_workspace_t* _data
