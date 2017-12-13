#pragma once

#include <math.h>
#include "../grid.h"


void integrate_rmin_rmax_o5(int n, sh_grid_t const* grid, double const f[grid->n[iR]], double res[grid->n[iR]]);

void integrate_rmin_rmax_o3(int n, sh_grid_t const* grid, double const f[grid->n[iR]], double res[grid->n[iR]]);
