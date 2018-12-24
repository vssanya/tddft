#pragma once

#include <math.h>
#include "../grid.h"

template <typename Grid>
double Dn_func(int n, Grid const& grid, int ir, int irl, double const* f) {
  return pow(grid.r(irl), n)/pow(grid.r(ir), n+1)*f[irl]*grid.J(irl, 0);
}

template <typename Grid>
double Un_func(int n, Grid const& grid, int ir, int irl, double const* f) {
  return pow(grid.r(ir), n)/pow(grid.r(irl), n+1)*f[irl]*grid.J(irl, 0);
}

template <typename Grid>
void integrate_rmin_rmax_o5(int n, Grid const& grid, double const* f, double* res);

template <typename Grid>
void integrate_rmin_rmax_o3(int n, Grid const& grid, double const* f, double* res);
