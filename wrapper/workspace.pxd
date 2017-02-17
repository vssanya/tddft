from types cimport cdouble, sphere_pot_t

from grid cimport sh_grid_t, sp_grid_t
from wavefunc cimport sphere_wavefunc_t

from field cimport field_t
from orbitals cimport ks_orbitals_t

cdef extern from "sphere_kn.h":
    ctypedef struct sphere_kn_workspace_t:
        double dt
        sh_grid_t* grid
        sphere_pot_t U
        sphere_pot_t Uabs
        cdouble* b
        cdouble* f
        cdouble* alpha
        cdouble* betta

    ctypedef struct sphere_kn_orbs_workspace_t:
        sphere_kn_workspace_t* wf_ws;
        double* Uh;
        double* Uxc;
        sp_grid_t* sp_grid;

    sphere_kn_workspace_t* sphere_kn_workspace_alloc(
            sh_grid_t* grid,
            double dt,
            sphere_pot_t U,
            sphere_pot_t Uabs
            )

    void sphere_kn_workspace_free(sphere_kn_workspace_t* ws)

    void sphere_kn_workspace_prop_ang(
            sphere_kn_workspace_t* ws,
            sphere_wavefunc_t* wf,
            int l, double E)

    void sphere_kn_workspace_prop_at(
            sphere_kn_workspace_t* ws,
            sphere_wavefunc_t* wf)

    void sphere_kn_workspace_prop_at_v2(
            sphere_kn_workspace_t* ws,
            sphere_wavefunc_t* wf,
            sphere_pot_t Ul,
            sphere_pot_t Uabs,
            bint img_time)

    void sphere_kn_workspace_prop(
            sphere_kn_workspace_t* ws,
            sphere_wavefunc_t* wf,
            field_t E,
            double t
            )

    void sphere_kn_workspace_prop_img(
            sphere_kn_workspace_t* ws,
            sphere_wavefunc_t* wf);

    void sphere_kn_workspace_orbs_prop_img(sphere_kn_workspace_t* ws, ks_orbitals_t* orbs)

    sphere_kn_orbs_workspace_t* sphere_kn_orbs_workspace_alloc(sh_grid_t* grid, double dt, sphere_pot_t U, sphere_pot_t Uabs)

    void sphere_kn_orbs_workspace_free(sphere_kn_orbs_workspace_t* ws);

    void sphere_kn_orbs_workspace_prop_img(sphere_kn_orbs_workspace_t* ws, ks_orbitals_t* orbs);


cdef class SKnWorkspace:
    cdef:
        sphere_kn_workspace_t* data

cdef class SOrbsWorkspace:
    cdef:
        sphere_kn_orbs_workspace_t* _data
