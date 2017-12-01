#include "sphere_harmonics.h"

#include <boost/math/special_functions/bessel.hpp>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>

#include "integrate.h"
#include <stdio.h>


jl_cache_t::jl_cache_t(sp_grid_t const* grid, int l_max):
	l_max(l_max),
	grid(grid)
{
	data = new double[grid->n[iR]*l_max]();
	for (int il=0; il<l_max; il++) {
		for (int ir=0; ir<grid->n[iR]; ir++) {
			(*this)(ir, il) = boost::math::sph_bessel(il, sp_grid_r(grid, ir));
		}
	}
}

jl_cache_t::~jl_cache_t() {
	delete[] data;
}

double jl_cache_t::operator()(double r, int il) const {
	int ir = sp_grid_ir(grid, r);
	double x = (r - sp_grid_r(grid, ir))/grid->d[iR];
	return (*this)(ir, il)*(1.0 - x) + (*this)(ir+1, il)*x;
}


int pow_minus_one(int p) {
	return p % 2 == 0 ? 1 : -1;
}

double clebsch_gordan_coef(int j1, int m1, int j2, int m2, int J, int M) {
	return pow_minus_one(j1-j2+M)*sqrt(2*J + 1)*gsl_sf_coupling_3j(2*j1, 2*j2, 2*J, 2*m1, 2*m2, -2*M);
}

double y3(int l1, int m1, int l2, int m2, int L, int M) {
	return sqrt((2*l1 + 1)*(2*l2 + 1)/(4*M_PI*(2*L + 1)))*clebsch_gordan_coef(l1, 0, l2, 0, L, 0)*clebsch_gordan_coef(l1, m1, l2, m2, L, M);
}

ylm_cache_t::ylm_cache_t(sp_grid_t const* grid, int l_max):
	l_max(l_max),
	grid(grid)
{
	size = gsl_sf_legendre_array_n(l_max);
	data = new double[2*(l_max+1)*grid->n[iC]]();

	double* tmp = new double[size]();
	for (int ic=0; ic<grid->n[iC]; ++ic) {
		double x = sp_grid_c(grid, ic);
		gsl_sf_legendre_array(GSL_SF_LEGENDRE_SPHARM, l_max, x, tmp);
		for (int m=0; m<2; ++m) {
			for (int l=0; l<=l_max; ++l) {
				(*this)(l, m, ic) = tmp[gsl_sf_legendre_array_index(l, m)];
			}
		}
	}

	delete[] tmp;
}

ylm_cache_t::~ylm_cache_t() {
	delete[] data;
}

ylm_cache_t* ylm_cache_new(int l_max, sp_grid_t const* grid) {
	return new ylm_cache_t(grid, l_max);
}

void ylm_cache_del(ylm_cache_t* cache) {
	delete cache;
}

double ylm_cache_get(ylm_cache_t const* cache, int l, int m, int ic) {
	return (*cache)(l, m, ic);
}

double ylm_cache_t::operator()(int l, int m, double c) const {
	int ic = sp_grid_ic(grid, c);
	double x = (c - sp_grid_c(grid, ic))/grid->d[iC];
	return (*this)(l, m, ic)*(1.0 - x) + (*this)(l, m, ic+1)*x;
}

double ylm_cache_calc(ylm_cache_t const* cache, int l, int m, double c) {
	return (*cache)(l, m, c);
}

double sh_series_r(func_2d_t f, int ir, int l, int m, sp_grid_t const* grid, ylm_cache_t const* ylm_cache) {
	return integrate_1d_cpp<double>([f, ir, ylm_cache, l, m](int ic) -> double {
			return f(ir, ic)*ylm_cache_get(ylm_cache, l, m, ic);
			}, grid->n[iC], grid->d[iC])*2*M_PI;
}

void sh_series(func_2d_t f, int l, int m, sp_grid_t const* grid, double* series, ylm_cache_t const* ylm_cache) {
#pragma omp parallel for
	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		series[ir] = sh_series_r(f, ir, l, m, grid, ylm_cache);
	}
}

