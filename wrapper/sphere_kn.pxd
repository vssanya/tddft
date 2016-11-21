from types cimport cdouble, sphere_pot_t, field_t

from sphere_grid cimport sphere_grid_t
from sphere_wavefunc cimport sphere_wavefunc_t

cdef extern from "sphere_kn.h":
    ctypedef struct sphere_kn_workspace_t:
        pass

    sphere_kn_workspace_t* sphere_kn_workspace_alloc(
        sphere_grid_t* grid,
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
