#include "sphere_harmonics.h"

#include <boost/math/special_functions/bessel.hpp>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>

#include "integrate.h"
#include <stdio.h>


JlCache::JlCache(SpGrid const* grid, int l_max):
	l_max(l_max),
	grid(grid)
{
	data = new double[(grid->n[iR]+1)*l_max]();
#pragma omp parallel for collapse(2)
	for (int il=0; il<l_max; il++) {
        for (int ir=-1; ir<grid->n[iR]; ir++) {
			if (ir == -1) {
				(*this)(ir, il) = boost::math::sph_bessel(il, 0.0);
			} else {
                (*this)(ir, il) = boost::math::sph_bessel(il, grid->r(ir));
			}
		}
	}
}

double JlCache::calc(double r, int il) {
    return boost::math::sph_bessel(il, r);
}

JlCache::~JlCache() {
	delete[] data;
}

double JlCache::operator()(double r, int il) const {
	assert(il >= 0 && il < l_max);

    int ir = grid->ir(r);
    double x = (r - grid->r(ir))/grid->d[iR];
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

YlmCache::YlmCache(SpGrid const* grid, int l_max):
	l_max(l_max),
	grid(grid)
{
	size = gsl_sf_legendre_array_n(l_max);
	data = new double[2*(l_max+1)*grid->n[iC]]();

	double* tmp = new double[size]();
	for (int ic=0; ic<grid->n[iC]; ++ic) {
        double x = grid->c(ic);
		gsl_sf_legendre_array(GSL_SF_LEGENDRE_SPHARM, l_max, x, tmp);
		for (int m=0; m<2; ++m) {
			for (int l=0; l<=l_max; ++l) {
				(*this)(l, m, ic) = tmp[gsl_sf_legendre_array_index(l, m)];
			}
		}
	}

	delete[] tmp;
}

YlmCache::~YlmCache() {
	delete[] data;
}

double YlmCache::operator()(int l, int m, double c) const {
    int ic = grid->ic(c);
    double x = (c - grid->c(ic))/grid->d[iC];

	if (ic == grid->n[iC] - 1) {
		return (*this)(l, m, ic);
	} else {
		return (*this)(l, m, ic)*(1.0 - x) + (*this)(l, m, ic+1)*x;
	}
}

double sh_series_r(std::function<double(int, int)> f, int ir, int l, int m, SpGrid const* grid, YlmCache const* ylm_cache) {
	return integrate_1d_cpp<double>([f, ir, ylm_cache, l, m](int ic) -> double {
            return f(ir, ic)*(*ylm_cache)(l, m, ic);
			}, grid->n[iC], grid->d[iC])*2*M_PI;
}

void sh_series(std::function<double(int, int)> f, int l, int m, SpGrid const* grid, double* series, YlmCache const* ylm_cache) {
#pragma omp parallel for
	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		series[ir] = sh_series_r(f, ir, l, m, grid, ylm_cache);
	}
}

