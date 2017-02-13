#pragma once

#include "grid.h"

// Potential
double hydrogen_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double hydrogen_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));

#include "sphere_wavefunc.h"
void hydrogen_ground(sphere_wavefunc_t* wf);
