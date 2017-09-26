#include "hp.h"
#include "../grid.h"
#include "../integrate.h"


double Dn_0(int n, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  double func(int irl) {
    return Dn_func(n, grid, 0, irl, f);
  }
  return (5/4*0.0 + 2*func(0) - func(1)/4)*grid->d[iR]/3;
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
    if (irl > grid->n[iR]-1) {
      return 0.0;
    }

    return Dn_func(n, grid, ir, irl, f);
  }

  double res;
  if (ir % 2 == 0) {
    res = pow((ir-1)/(double)(ir+1), n+1)*F[0] + (func(ir-2) + 4*func(ir-1) + func(ir))*grid->d[iR]/3;
  } else {
    res = pow(    ir/(double)(ir+1), n+1)*F[1] + (5*func(ir-1)/4 + 2*func(ir) - func(ir+1)/4)*grid->d[iR]/3;
  }

  F[0] = F[1];
  F[1] = res;

  return res;
}

double Un_0(int n, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  return 0.0;
}

double Un_1(int n, sh_grid_t const* grid, double const f[grid->n[iR]]) {
  int ir = grid->n[iR]-2;

  double func(int irl) {
    return Un_func(n, grid, ir, irl, f);
  }

  return (func(ir) + func(ir+1))*grid->d[iR]*0.5;
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
    if (irl < 0) {
      return 0.0;
    }

    return Un_func(n, grid, ir, irl, f);
  }

  double res;
  if (ir % 2 == 0) {
    res = pow((ir+1)/(double)(ir+3), n)*F[0] + (func(ir) + 4*func(ir+1) + func(ir+2))*grid->d[iR]/3;
  } else {
    res = pow((ir+1)/(double)(ir+2), n)*F[1] + (-func(ir-1)/4 + 2*func(ir) + 5*func(ir+1)/4)*grid->d[iR]/3;
  }

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
