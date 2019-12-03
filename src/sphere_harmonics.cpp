#include "sphere_harmonics.h"

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_bessel.h>

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
	return pow_minus_one(-j1+j2-M)*sqrt(2*J + 1)*gsl_sf_coupling_3j(2*j1, 2*j2, 2*J, 2*m1, 2*m2, -2*M);
}

double y3(int l1, int m1, int l2, int m2, int L, int M) {
	return sqrt((2*l1 + 1)*(2*l2 + 1)/(4*M_PI*(2*L + 1)))*clebsch_gordan_coef(l1, 0, l2, 0, L, 0)*clebsch_gordan_coef(l1, m1, l2, m2, L, M);
}

YlmCache::YlmCache(SpGrid const* grid, int l_max):
	l_max(l_max),
	grid(grid)
{
	data = new double[2*(l_max+1)*grid->n[iC]]();
	for (int ic=0; ic<grid->n[iC]; ++ic) {
        double theta = grid->theta(ic);
		for (int m=0; m<2; ++m) {
			for (int l=0; l<=l_max; ++l) {
                (*this)(l, m, ic) = YlmCache::calc(l, m, theta);
			}
		}
	}
}

YlmCache::~YlmCache() {
	delete[] data;
}

double YlmCache::calc(int l, int m, double theta) {
    return boost::math::spherical_harmonic_r(l, m, theta, 0.0);
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

