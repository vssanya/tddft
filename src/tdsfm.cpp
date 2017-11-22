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

#include <boost/math/special_functions/bessel.hpp>

#include <stdlib.h>
#include <stdio.h>

#include "tdsfm.h"
#include "integrate.h"


tdsfm_t* tdsfm_new(sp_grid_t const* k_grid, sh_grid_t const* r_grid, double A_max, int ir) {
	tdsfm_t* tdsfm = new tdsfm_t;
	tdsfm->k_grid = k_grid;
	tdsfm->r_grid = r_grid;
	tdsfm->ir = ir;
	tdsfm->data = new cdouble[k_grid->n[iR]*k_grid->n[iC]]();

	double k_max = sp_grid_r(k_grid, k_grid->n[iR]-1) + A_max;
	int N[3] = {(int)(k_max/k_grid->d[iR]), k_grid->n[1], k_grid->n[2]};
	tdsfm->appr_k_grid = sp_grid_new(N, k_max);
	tdsfm->jl = new double[tdsfm->appr_k_grid->n[iR]*(r_grid->n[iL]+1)];
	double r = sh_grid_r(r_grid, ir);
	for (int ik=0; ik<k_grid->n[iR]; ik++) {
		double k = sp_grid_r(k_grid, ik);
		for (int il=0; il<(r_grid->n[iL]+1); il++) {
			tdsfm->jl[il + (r_grid->n[iL]+1)*ik] = boost::math::sph_bessel(il, k*r);
		}
	}

	tdsfm->ylm = ylm_cache_new(r_grid->n[iL], tdsfm->appr_k_grid);

	tdsfm->int_A  = 0.0;
	tdsfm->int_A2 = 0.0;

	return tdsfm;
}

void tdsfm_del(tdsfm_t* tdsfm) {
	ylm_cache_del(tdsfm->ylm);
	delete[] tdsfm->jl;
	delete[] tdsfm->data;
	sp_grid_del(tdsfm->appr_k_grid);
	delete tdsfm;
}

void tdsfm_calc(tdsfm_t* tdsfm, field_t const* field, sh_wavefunc_t const* wf, double t, double dt) {
	int Nk = tdsfm->k_grid->n[iR];
	int Nl = tdsfm->r_grid->n[iL];

	double ir = tdsfm->ir;
	double r = sh_grid_r(wf->grid, ir);
	double dr = wf->grid->d[iR];

	double At_dt = field_A(field, t-dt);
	double At    = field_A(field, t);
	tdsfm->int_A  += (At + At_dt)*dt*0.5;
	tdsfm->int_A2 += (pow(At, 2) + pow(At_dt, 2))*dt*0.5;

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<Nk; ik++) {
		for (int ic=0; ic<tdsfm->k_grid->n[iC]; ic++) {
			double k = sp_grid_r(tdsfm->k_grid, ik);
			double kz = sp_grid_c(tdsfm->k_grid, ic)*k;

			cdouble S = cexp(0.5*I*(k*k*t - 2*kz*tdsfm->int_A + tdsfm->int_A2));

			double k_A = sqrt(k*k + 2*kz*At + At*At);
			double k_A_z = kz + At;

			int ik_A   = sp_grid_ir(tdsfm->appr_k_grid, k_A);
			int ic_k_A = sp_grid_ic(tdsfm->appr_k_grid, k_A_z/k_A);

			cdouble a_k = 0.0;
			for (int il=0; il<wf->grid->n[iL]; il++) {
				cdouble psi = swf_get(wf, ir, il);
				cdouble dpsi = (swf_get(wf, ir+1, il) - swf_get(wf, ir-1, il))/(2*dr);
				a_k += cpow(-I, il+1)*(
						tdsfm->jl[il  +ik_A*(Nl+1)]*(dpsi-(il+1)*psi/r) +
						tdsfm->jl[il+1+ik_A*(Nl+1)]*k_A*psi
						)*ylm_cache_get(tdsfm->ylm, il, wf->m, ic_k_A);
			}

			tdsfm->data[ik+ic*Nk] += a_k*r/sqrt(2.0*M_PI)*S*dt;
		}
	}
}

void tdsfm_calc_inner(tdsfm_t* tdsfm, field_t const* field, sh_wavefunc_t const* wf, double t, int ir_min, int ir_max) {
	int Nk = tdsfm->k_grid->n[iR];

	double At    = field_A(field, t);
	double Az  = t*At - tdsfm->int_A;
	double A_2 = t*pow(At, 2) - 2*At*tdsfm->int_A + tdsfm->int_A2;

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<Nk; ik++) {
		for (int ic=0; ic<tdsfm->k_grid->n[iC]; ic++) {
			double k = sp_grid_r(tdsfm->k_grid, ik);
			double kz = sp_grid_c(tdsfm->k_grid, ic)*k;

			cdouble S = cexp(0.5*I*(k*k*t + 2*kz*Az + A_2));

			cdouble a_k = 0.0;
			for (int il=0; il<wf->grid->n[iL]; il++) {
				cdouble a_kl = 0.0;
				for (int ir=ir_min; ir<ir_max; ir++) {
					double r = sh_grid_r(wf->grid, ir);
					a_kl += r*swf_get(wf, ir, il)*boost::math::sph_bessel(il, k*r);
				}
				a_k += a_kl*cpow(-I, il)*ylm_cache_get(tdsfm->ylm, il, wf->m, ic);
			}

			tdsfm->data[ik+ic*Nk] += a_k*sqrt(2.0/M_PI)*wf->grid->d[iR]*S;
		}
	}
}

double jn(int l, double x) {
	return boost::math::sph_bessel(l, x);
}
