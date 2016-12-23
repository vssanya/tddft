from types cimport cdouble, sphere_pot_t, sphere_pot_abs_t

from grid cimport sh_grid_t
from wavefunc cimport sphere_wavefunc_t

from field cimport field_t

cdef extern from "sphere_kn.h":
    ctypedef struct sphere_kn_workspace_t:
        pass

    sphere_kn_workspace_t* sphere_kn_workspace_alloc(
        sh_grid_t* grid,
        double dt,
        sphere_pot_t U,
        sphere_pot_abs_t Uabs
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
        sphere_wavefunc_t* wf
    )

    void sphere_kn_workspace_prop(
        sphere_kn_workspace_t* ws,
        sphere_wavefunc_t* wf,
        field_t E,
        double t
    )

cdef class SKnWorkspace:
    cdef:
        sphere_kn_workspace_t* data
