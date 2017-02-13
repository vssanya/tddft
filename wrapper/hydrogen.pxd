from wavefunc cimport sphere_wavefunc_t
from grid cimport sh_grid_t

cdef extern from "hydrogen.h":
    double hydrogen_sh_u(sh_grid_t* grid, int ir, int il, int m)
    double hydrogen_sh_dudz(sh_grid_t* grid, int ir, int il, int m)
    void hydrogen_ground(sphere_wavefunc_t* wf)
