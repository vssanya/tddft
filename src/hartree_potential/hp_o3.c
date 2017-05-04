#include "hp.h"
#include "../integrate.h"


double Dn_func_o3(int n, sh_grid_t const* grid, int ir, int irl, double const f[grid->n[iR]]) {
  return pow(sh_grid_r(grid, irl), n)/pow(sh_grid_r(grid, ir), n+1)*f[irl];
}

double Dn_0_o3(int n, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  double func(int irl) {
    return Dn_func_o3(n, grid, 0, irl, f);
  }
  return (0.0 + func(0))*grid->d[iR]*0.5;
}

void Dn_init_o3(int n, double F[1], sh_grid_t const* grid, double const f[grid->n[iR]]) {
  F[0] = Dn_0_o3(n, grid, f);
}

double Dn_next_o3(int n, double F[1], int ir, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  if (ir == 0) {
    return F[ir];
  }

  double func(int irl) {
    return Dn_func_o3(n, grid, ir, irl, f);
  }

  double res = pow(ir/(double)(ir+1), n+1)*F[0] + (func(ir-1) + func(ir))*grid->d[iR]*0.5;

  F[0] = res;
  return res;
}

double Un_func_o3(int n, sh_grid_t const* grid, int ir, int irl, double const f[grid->n[iR]]) {
  return pow(sh_grid_r(grid, ir), n)/pow(sh_grid_r(grid, irl), n+1)*f[irl];
}

double Un_0_o3(int n, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  return 0.0;
}

void Un_init_o3(int n, double F[1], sh_grid_t const* grid, double const f[grid->n[iR]]) {
  F[0] = Un_0_o3(n, grid, f);
}

double Un_next_o3(int n, double F[1], int ir, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  if (ir == grid->n[iR]-1) {
    return F[grid->n[iR] - 1 - ir];
  }

  double func(int irl) {
    return Un_func_o3(n, grid, ir, irl, f);
  }

  double res = pow((ir+1)/(double)(ir+2), n)*F[0] + (func(ir) + func(ir+1))*grid->d[iR]*0.5;

  F[0] = res;

  return res;
}

void integrate_rmin_rmax_o3(int n, sh_grid_t const* grid, double const f[grid->n[iR]], double res[grid->n[iR]]) {
  double F[1];

  Dn_init_o3(n, F, grid, f);
  for (int ir = 0; ir < grid->n[iR]; ++ir) {
    res[ir] = Dn_next_o3(n, F, ir, grid, f);
  }

  Un_init_o3(n, F, grid, f);
  for (int ir = grid->n[iR] - 1; ir >= 0; --ir) {
    res[ir] += Un_next_o3(n, F, ir, grid, f);
  }
}

/* void integrate_rmin_rmax_o3(int n, sh_grid_t const* grid, double const f[grid->n[iR]], double res[grid->n[iR]]) { */
/*   double F[2]; */

/*   res[0] = F_first(F, n, grid, f); */
/*   for (int ir = 0; ir < grid->n[iR]; ++ir) { */
/* 		res[ir] = F_next(F, n, ir, grid, f); */
/* 	} */
/* } */
