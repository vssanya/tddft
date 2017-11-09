#pragma once

#include <math.h>
#include "../grid.h"


inline double Dn_func(int n, sh_grid_t const* grid, int ir, int irl, double const f[grid->n[iR]]) {
  return pow(sh_grid_r(grid, irl), n)/pow(sh_grid_r(grid, ir), n+1)*f[irl];
}

inline double Un_func(int n, sh_grid_t const* grid, int ir, int irl, double const f[grid->n[iR]]) {
  return pow(sh_grid_r(grid, ir), n)/pow(sh_grid_r(grid, irl), n+1)*f[irl];
}

void integrate_rmin_rmax_o5(int n, sh_grid_t const* grid, double const f[grid->n[iR]], double res[grid->n[iR]]);

void integrate_rmin_rmax_o3(int n, sh_grid_t const* grid, double const f[grid->n[iR]], double res[grid->n[iR]]);
