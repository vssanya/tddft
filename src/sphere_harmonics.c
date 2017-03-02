#include "sphere_harmonics.h"

#include <stddef.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>


int pow_minus_one(int p) {
	return p % 2 == 0 ? 1 : -1;
}

double clebsch_gordan_coef(int j1, int m1, int j2, int m2, int J, int M) {
	return pow_minus_one(j1-j2+M)*sqrt(2*J + 1)*gsl_sf_coupling_3j(2*j1, 2*j2, 2*J, 2*m1, 2*m2, -2*M);
}

double y3(int l1, int m1, int l2, int m2, int L, int M) {
	return sqrt((2*l1 + 1)*(2*l2 + 1)/(4*M_PI*(2*L + 1)))*clebsch_gordan_coef(l1, 0, l2, 0, L, 0)*clebsch_gordan_coef(l1, m1, l2, m2, L, M);
}

static struct {
	double* data;
	int size;
	int l_max;
	sp_grid_t const* grid;
	int ref;
} ylm_cache;

void ylm_cache_init(int l_max, sp_grid_t const* grid) {
	if (ylm_cache.l_max < l_max) {
		ylm_cache.l_max = l_max;
		ylm_cache.grid = grid;
		ylm_cache.size = gsl_sf_legendre_array_n(l_max);

		if (ylm_cache.data != NULL) {
			free(ylm_cache.data);
		}

		ylm_cache.data = malloc(sizeof(double)*ylm_cache.size*grid->n[iC]);
		for (int ic=0; ic<grid->n[iC]; ++ic) {
			double x = sp_grid_c(grid, ic);
			gsl_sf_legendre_array(GSL_SF_LEGENDRE_SPHARM, l_max, x, &ylm_cache.data[ylm_cache.size*ic]);
		}
	}

	ylm_cache.ref++;
}

void ylm_cache_deinit() {
	ylm_cache.ref--;
	if (ylm_cache.ref == 0) {
		free(ylm_cache.data);
	}
}

double ylm(int l, int m, int ic) {
	assert(ylm_cache.data != NULL);
	assert(l <= ylm_cache.l_max);

	return ylm_cache.data[gsl_sf_legendre_array_index(l, m) + ic*ylm_cache.size];
}

double sh_series_r(func_2d_t f, int ir, int l, int m, sp_grid_t const* grid) {
	double func(int ic) {
		return f(ir, ic)*ylm(l, m, ic);
	}

	return integrate_1d(func, grid->n[iC], grid->d[iC])*2*M_PI;
}

void sh_series(func_2d_t f, int l, int m, sp_grid_t const* grid, double series[grid->n[iR]]) {
#pragma omp parallel for
	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		series[ir] = sh_series_r(f, ir, l, m, grid);
	}
}

