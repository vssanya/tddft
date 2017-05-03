#include "hp.h"
#include "../grid.h"
#include "../integrate.h"


/*!
 * \breif \f[D_n(r) = \int_0^r \frac{r'^n}{r^{n+1}}f(r')dr'\f]
 * */
double Dn_func(int n, sh_grid_t const* grid, int ir, int irl, double const f[grid->n[iR]]) {
  return pow(sh_grid_r(grid, irl), n)/pow(sh_grid_r(grid, ir), n+1)*f[irl];
}

double Dn_0(int n, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  double func(int irl) {
    return Dn_func(n, grid, 0, irl, f);
  }
  return (0.0 + func(0))*grid->d[iR]*0.5;
}

double Dn_1(int n, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  double func(int irl) {
    return Dn_func(n, grid, 1, irl, f);
  }

  return (0.0 + 4*func(0) + func(1))*grid->d[iR]/3;
}

void Dn_init(int n, double F[2], sh_grid_t const* grid, double const f[grid->n[iR]]) {
  F[0] = Dn_0(n, grid, f);
  F[1] = Dn_1(n, grid, f);
}

double Dn_next(int n, double F[2], int ir, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  if (ir <= 1) {
    return F[ir];
  }

  double func(int irl) {
    return Dn_func(n, grid, ir, irl, f);
  }

  double res = pow((ir-2)/(double)ir, n+1)*F[0] + (func(ir-2) + 4*func(ir-1) + func(ir))*grid->d[iR]/3;

  F[0] = F[1];
  F[1] = res;

  return res;
}

double Un_func(int n, sh_grid_t const* grid, int ir, int irl, double const f[grid->n[iR]]) {
  return pow(sh_grid_r(grid, ir), n)/pow(sh_grid_r(grid, irl), n+1)*f[irl];
}

double Un_0(int n, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  return 0.0;
}

double Un_1(int n, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  double func(int irl) {
    return Un_func(n, grid, grid->n[iR]-2, irl, f);
  }

  return (func(grid->n[iR]-2) + func(grid->n[iR]-1))*grid->d[iR]*0.5;
}

void Un_init(int n, double F[2], sh_grid_t const* grid, double const f[grid->n[iR]]) {
  F[0] = Un_0(n, grid, f);
  F[1] = Un_1(n, grid, f);
}

double Un_next(int n, double F[2], int ir, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  if (ir >= grid->n[iR]-2) {
    return F[grid->n[iR] - 1 - ir];
  }

  double func(int irl) {
    return Un_func(n, grid, ir, irl, f);
  }

  double res = pow((ir)/(double)(ir+2), n)*F[0] + (func(ir) + 4*func(ir+1) + func(ir+2))*grid->d[iR]/3;

  F[0] = F[1];
  F[1] = res;

  return res;
}

void integrate_rmin_rmax_o5(int n, sh_grid_t const* grid, double const f[grid->n[iR]], double res[grid->n[iR]]) {
  double F[2];

  Dn_init(n, F, grid, f);
  for (int ir = 0; ir < grid->n[iR]; ++ir) {
    res[ir] = Dn_next(n, F, ir, grid, f);
  }

  Un_init(n, F, grid, f);
  for (int ir = grid->n[iR] - 1; ir >= 0; --ir) {
    res[ir] += Un_next(n, F, ir, grid, f);
  }
}
