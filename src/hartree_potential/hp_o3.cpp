#include "hp.h"
#include "../integrate.h"

template <typename Grid>
double Dn_0_o3(int n, Grid const& grid, double const* f) {
  auto func = [n,&grid,f](int irl) -> double {
    return Dn_func(n, grid, 0, irl, f);
  };
  return (0.0 + func(0))*grid.d[iR]*0.5;
}

template <typename Grid>
void Dn_init_o3(int n, double F[1], Grid const& grid, double const* f) {
  F[0] = Dn_0_o3(n, grid, f);
}

template <typename Grid>
double Dn_next_o3(int n, double F[1], int ir, Grid const& grid, double const* f) {
  if (ir == 0) {
    return F[0];
  }

  auto func = [n, &grid, ir, f](int irl) -> double {
    return Dn_func(n, grid, ir, irl, f);
  };

  double res = pow(grid.r(ir-1)/grid.r(ir), n+1)*F[0] + (func(ir-1) + func(ir))*grid.d[iR]*0.5;

  F[0] = res;
  return res;
}

template <typename Grid>
double Un_0_o3(int n, Grid const& grid, double const* f) {
  return 0.0;
}

template <typename Grid>
void Un_init_o3(int n, double F[1], Grid const& grid, double const* f) {
  F[0] = Un_0_o3(n, grid, f);
}

template <typename Grid>
double Un_next_o3(int n, double F[1], int ir, Grid const& grid, double const* f) {
  if (ir == grid.n[iR]-1) {
    return F[0];
  }

  auto func = [n,&grid,ir,f](int irl) -> double {
    return Un_func(n, grid, ir, irl, f);
  };

  double res = pow(grid.r(ir)/grid.r(ir+1), n)*F[0] + (func(ir) + func(ir+1))*grid.d[iR]*0.5;

  F[0] = res;

  return res;
}

template <typename Grid>
void integrate_rmin_rmax_o3(int n, Grid const& grid, double const* f, double* res) {
  double F[1];

  Dn_init_o3(n, F, grid, f);
  for (int ir = 0; ir < grid.n[iR]; ++ir) {
    res[ir] = Dn_next_o3(n, F, ir, grid, f);
  }

  Un_init_o3(n, F, grid, f);
  for (int ir = grid.n[iR] - 1; ir >= 0; --ir) {
    res[ir] += Un_next_o3(n, F, ir, grid, f);
  }
}

/* void integrate_rmin_rmax_o3(int n, cGrid const& grid, double const f[grid.n[iR]], double res[grid.n[iR]]) { */
/*   double F[2]; */

/*   res[0] = F_first(F, n, grid, f); */
/*   for (int ir = 0; ir < grid.n[iR]; ++ir) { */
/* 		res[ir] = F_next(F, n, ir, grid, f); */
/* 	} */
/* } */

template
void integrate_rmin_rmax_o3<ShGrid>(int n, ShGrid const& grid, double const* f, double* res);

template
void integrate_rmin_rmax_o3<ShNotEqudistantGrid>(int n, ShNotEqudistantGrid const& grid, double const* f, double* res);
