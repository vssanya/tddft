/*
 * =====================================================================================
 *
 *       Filename:  tdsfm.c
 *
 *    Description:  Time-dependent surface flux method
 *
 *        Version:  1.0
 *        Created:  11.10.2017 14:59:40
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Romanov Alexander, 
 *   Organization:  IAP RAS
 *
 * =====================================================================================
 */

//#include <boost/math/special_functions/bessel.hpp>
//#include <boost/math/special_functions/spherical_harmonic.hpp>

#include <stdlib.h>
#include <stdio.h>

#include "tdsfm.h"
#include "integrate.h"


tdsfm_t::tdsfm_t(sp_grid_t const* k_grid, sh_grid_t const* r_grid, double A_max, int ir):
	k_grid(k_grid),
	r_grid(r_grid),
	ir(ir),
	int_A(0.0),
	int_A2(0.0)
{
	data = new cdouble[k_grid->n[iR]*k_grid->n[iC]]();

	double k_max = (sp_grid_r(k_grid, k_grid->n[iR]-1) + A_max);
	double r_max = sh_grid_r_max(r_grid);

	int N[3] = {(int)(k_max*r_max/(k_grid->d[iR]*r_grid->d[iR]))*2, k_grid->n[1]*2, k_grid->n[2]};
	jl_grid = sp_grid_new(N, k_max*r_max);
	jl = new jl_cache_t(jl_grid, r_grid->n[iL]+1);

	int N_ylm[3] = {(int)(k_max/k_grid->d[iR])*2, k_grid->n[1]*2, k_grid->n[2]};
	ylm_grid = sp_grid_new(N_ylm, k_max);
	ylm = new ylm_cache_t(ylm_grid, r_grid->n[iL]);
}

tdsfm_t::~tdsfm_t() {
	delete ylm;
	delete jl;
	delete[] data;
	sp_grid_del(jl_grid);
	sp_grid_del(ylm_grid);
}

void tdsfm_t::calc(field_t const* field, sh_wavefunc_t const& wf, double t, double dt) {
	int Nk = k_grid->n[iR];
	int Nl = r_grid->n[iL];

	double r = sh_grid_r(wf.grid, ir);
	double dr = wf.grid->d[iR];

	double At_dt = field_A(field, t-dt);
	double At    = field_A(field, t);

	int_A  += (At + At_dt)*dt*0.5;
	int_A2 += (pow(At, 2) + pow(At_dt, 2))*dt*0.5;

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<Nk; ik++) {
		for (int ic=0; ic<k_grid->n[iC]; ic++) {
			double k  = sp_grid_r(k_grid, ik);
			double kz = sp_grid_c(k_grid, ic)*k;

			cdouble S = cexp(0.5*I*(k*k*t + 2*kz*int_A + int_A2));

			double k_A = sqrt(k*k + 2*kz*At + At*At);
			double k_A_z = kz + At;

			cdouble a_k = 0.0;
			for (int il=0; il<wf.grid->n[iL]; il++) {
				cdouble psi = wf(ir, il);
				cdouble dpsi = (wf(ir+1, il) - wf(ir-1, il))/(2*dr);
				a_k += cpow(-I, il+1)*(
						(*jl)(k_A*r, il)*(dpsi-(il+1)*psi/r) +
						//boost::math::sph_bessel(il, k_A*r)*(dpsi-(il+1)*psi/r) +
						(*jl)(k_A*r, il+1)*k_A*psi
						//boost::math::sph_bessel(il+1, k_A*r)*k_A*psi
						)*(*ylm)(il, wf.m, k_A_z/k_A);
						//)*boost::math::spherical_harmonic_r(il, wf->m, acos(k_A_z/k_A), 0);
			}

			data[ik+ic*Nk] += a_k*r/sqrt(2.0*M_PI)*S*dt;
		}
	}
}

void tdsfm_t::calc_inner(field_t const* field, sh_wavefunc_t const& wf, double t, int ir_min, int ir_max) {
	int Nk = k_grid->n[iR];

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<Nk; ik++) {
		for (int ic=0; ic<k_grid->n[iC]; ic++) {
			double k = sp_grid_r(k_grid, ik);
			double kz = sp_grid_c(k_grid, ic)*k;

			cdouble S = cexp(0.5*I*(k*k*t - 2*kz*int_A + int_A2));

			cdouble a_k = 0.0;
			for (int il=0; il<wf.grid->n[iL]; il++) {
				cdouble a_kl = 0.0;
				for (int ir=ir_min; ir<ir_max; ir++) {
					double r = sh_grid_r(wf.grid, ir);
					a_kl += r*wf(ir, il)*(*jl)(il, k*r);
				}
				a_k += a_kl*cpow(-I, il)*(*ylm)(il, wf.m, ic);
			}

			data[ik+ic*Nk] += a_k*sqrt(2.0/M_PI)*wf.grid->d[iR]*S;
		}
	}
}
