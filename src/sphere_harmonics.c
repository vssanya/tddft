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

ylm_cache_t* ylm_cache_new(int l_max, sp_grid_t const* grid) {
	ylm_cache_t* cache = malloc(sizeof(ylm_cache_t));

	cache->l_max = l_max;
	cache->grid = grid;
	cache->size = gsl_sf_legendre_array_n(l_max);
	cache->data = malloc(sizeof(double)*2*(cache->l_max+1)*grid->n[iC]);

	double* tmp = malloc(sizeof(double)*cache->size);
	for (int ic=0; ic<grid->n[iC]; ++ic) {
		double x = sp_grid_c(grid, ic);
		gsl_sf_legendre_array(GSL_SF_LEGENDRE_SPHARM, l_max, x, tmp);
		for (int m=0; m<2; ++m) {
			for (int l=0; l<=cache->l_max; ++l) {
				cache->data[l + ic*(cache->l_max+1) + m*(cache->l_max+1)*grid->n[iC]] = tmp[gsl_sf_legendre_array_index(l, m)];
			}
		}
	}
	free(tmp);

	return cache;
}

void ylm_cache_del(ylm_cache_t* cache) {
	free(cache->data);
	free(cache);
}

double ylm_cache_get(ylm_cache_t const* cache, int l, int m, int ic) {
	assert(cache->data != NULL);
	assert(l <= cache->l_max);

	return cache->data[l + ic*(cache->l_max+1) + m*(cache->l_max+1)*cache->grid->n[iC]];
}

double sh_series_r(func_2d_t f, int ir, int l, int m, sp_grid_t const* grid, ylm_cache_t const* ylm_cache) {
	double func(int ic) {
		return f(ir, ic)*ylm_cache_get(ylm_cache, l, m, ic);
	}

	return integrate_1d(func, grid->n[iC], grid->d[iC])*2*M_PI;
}

void sh_series(func_2d_t f, int l, int m, sp_grid_t const* grid, double series[grid->n[iR]], ylm_cache_t const* ylm_cache) {
#pragma omp parallel for
	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		series[ir] = sh_series_r(f, ir, l, m, grid, ylm_cache);
	}
}

