from types cimport cdouble, sh_f

from grid cimport sh_grid_t, sp_grid_t
from abs_pot cimport uabs_sh_t, Uabs
from wavefunc cimport sh_wavefunc_t

from field cimport field_t
from orbitals cimport orbitals_t
from sphere_harmonics cimport ylm_cache_t
from atom cimport atom_t

cdef extern from "sh_workspace.h":
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
            field_t E,
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
            field_t field,
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
        sh_workspace_t* data
        Uabs uabs

cdef class SOrbsWorkspace:
    cdef:
        sh_orbs_workspace_t* _data
        Uabs uabs
