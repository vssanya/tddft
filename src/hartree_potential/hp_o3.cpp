#include "hp.h"
#include "../integrate.h"
#include "../types.h"

template <typename Grid, typename T>
T Dn_0_o3(int n, Grid const& grid, T const* f) {
  auto func = [n,&grid,f](int irl) -> T {
    return Dn_func(n, grid, 0, irl, f);
  };
  return (0.0 + func(0))*grid.d[iR]*0.5;
}

template <typename Grid, typename T>
void Dn_init_o3(int n, T F[1], Grid const& grid, T const* f) {
  F[0] = Dn_0_o3(n, grid, f);
}

template <typename Grid, typename T>
T Dn_next_o3(int n, T F[1], int ir, Grid const& grid, T const* f) {
  if (ir == 0) {
    return F[0];
  }

  auto func = [n, &grid, ir, f](int irl) -> T {
    return Dn_func(n, grid, ir, irl, f);
  };

  T res = pow(grid.r(ir-1)/grid.r(ir), n+1)*F[0] + (func(ir-1) + func(ir))*grid.d[iR]*0.5;

  F[0] = res;
  return res;
}

template <typename Grid, typename T>
T Un_0_o3(int n, Grid const& grid, T const* f) {
  return 0.0;
}

template <typename Grid, typename T>
void Un_init_o3(int n, T F[1], Grid const& grid, T const* f) {
  F[0] = Un_0_o3(n, grid, f);
}

template <typename Grid, typename T>
T Un_next_o3(int n, T F[1], int ir, Grid const& grid, T const* f) {
  if (ir == grid.n[iR]-1) {
    return F[0];
  }

  auto func = [n,&grid,ir,f](int irl) -> T {
    return Un_func(n, grid, ir, irl, f);
  };

  T res = pow(grid.r(ir)/grid.r(ir+1), n)*F[0] + (func(ir) + func(ir+1))*grid.d[iR]*0.5;

  F[0] = res;

  return res;
}

template <typename Grid, typename T>
void integrate_rmin_rmax_o3(int n, Grid const& grid, T const* f, T* res) {
  T F[1];

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
void integrate_rmin_rmax_o3<ShGrid, double>(int n, ShGrid const& grid, double const* f, double* res);
template
void integrate_rmin_rmax_o3<ShGrid, cdouble>(int n, ShGrid const& grid, cdouble const* f, cdouble* res);

template
void integrate_rmin_rmax_o3<ShNotEqudistantGrid, double>(int n, ShNotEqudistantGrid const& grid, double const* f, double* res);
template
void integrate_rmin_rmax_o3<ShNotEqudistantGrid, cdouble>(int n, ShNotEqudistantGrid const& grid, cdouble const* f, cdouble* res);
