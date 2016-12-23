#include "sphere_harmonics.h"

#include <stddef.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>

int pow_minus_one(int p) {
	return p % 2 == 0 ? 1 : -1;
}

double clebsch_gordan_coef(int j1, int m1, int j2, int m2, int J, int M) {
	return pow_minus_one(j1-j2+M)*sqrt(2*J + 1)*gsl_sf_coupling_3j(j1, j2, J, m1, m2, -M);
}

double y3(int l1, int m1, int l2, int m2, int L, int M) {
	return sqrt((2*l1 + 1)*(2*l2 + 1)/(4*M_PI*(2*L + 1)))*clebsch_gordan_coef(l1, 0, l2, 0, L, 0)*clebsch_gordan_coef(l1, m1, l2, m2, L, M);
}

double Ylm(int l, int m, double x) {
	double res[gsl_sf_legendre_array_n(l)];
	int err = gsl_sf_legendre_array(GSL_SF_LEGENDRE_SPHARM, l, x, res);
	return res[gsl_sf_legendre_array_index(l, m)];
}

void sh_series(func_2d_t f, int l, int m, sp_grid_t const* grid, double series[grid->n[iR]]) {
	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		double func(int i) {
			double x = sp_grid_c(grid, i);
			return f(ir, i)*Ylm(l, m, x);
		}

		series[ir] = integrate_1d(func, grid->n[iC], grid->d[iC])*2*M_PI;
	}
}
