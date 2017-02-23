from types cimport cdouble, sh_f

from grid cimport sh_grid_t, sp_grid_t
from wavefunc cimport sphere_wavefunc_t

from field cimport field_t
from orbitals cimport ks_orbitals_t

cdef extern from "sh_workspace.h":
    ctypedef struct sh_workspace_t:
        sh_grid_t* grid
        sh_f U
        sh_f Uabs
        cdouble* b
        cdouble* f
        cdouble* alpha
        cdouble* betta

    ctypedef struct sh_orbs_workspace_t:
        sh_workspace_t* wf_ws;
        double* Uh;
        double* Uxc;
        sp_grid_t* sp_grid;

    sh_workspace_t* sh_workspace_alloc(
            sh_grid_t* grid,
            sh_f U,
            sh_f Uabs
            )

    void sh_workspace_free(sh_workspace_t* ws)

    void sh_workspace_prop_ang(
            sh_workspace_t* ws,
            sphere_wavefunc_t* wf,
            double dt,
            int l, double E)

    void sh_workspace_prop_at(
            sh_workspace_t* ws,
            sphere_wavefunc_t* wf,
            cdouble dt,
            sh_f Ul,
            sh_f Uabs
    )

    void sh_workspace_prop_at_v2(
            sh_workspace_t* ws,
            sphere_wavefunc_t* wf,
            cdouble dt,
            sh_f Ul,
            sh_f Uabs
    )

    void sh_workspace_prop(
            sh_workspace_t* ws,
            sphere_wavefunc_t* wf,
            field_t E,
            double t,
            double dt
    )

    void sh_workspace_prop_img(
            sh_workspace_t* ws,
            sphere_wavefunc_t* wf,
            double dt
    )

    void sh_orbs_workspace_prop(
            sh_orbs_workspace_t* ws,
            ks_orbitals_t* orbs,
            field_t field,
            double t,
            double dt
    )
    void sh_orbs_workspace_prop_img(
            sh_orbs_workspace_t* ws,
            ks_orbitals_t* orbs,
            double dt
    )

    sh_orbs_workspace_t* sh_orbs_workspace_alloc(
            sh_grid_t* grid,
            sh_f U,
            sh_f Uabs
    )
    void sh_orbs_workspace_free(sh_orbs_workspace_t* ws)


cdef class SKnWorkspace:
    cdef:
        sh_workspace_t* data

cdef class SOrbsWorkspace:
    cdef:
        sh_orbs_workspace_t* _data
