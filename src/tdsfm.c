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

#include <stdlib.h>
#include <gsl/gsl_sf_bessel.h>

#include "tdsfm.h"
#include "integrate.h"


tdsfm_t* tdsfm_new(sp_grid_t const* k_grid, sh_grid_t const* r_grid, int ir) {
	tdsfm_t* tdsfm = malloc(sizeof(tdsfm_t));
	tdsfm->k_grid = k_grid;
	tdsfm->r_grid = r_grid;
	tdsfm->ir = ir;
	tdsfm->data = calloc(k_grid->n[iR]*k_grid->n[iC], sizeof(cdouble));

	tdsfm->jl = calloc(k_grid->n[iR]*(r_grid->n[iL]+1), sizeof(double));
	double r = sh_grid_r(r_grid, ir);
	for (int ik=0; ik<k_grid->n[iR]; ik++) {
		double k = sp_grid_r(k_grid, ik);
		gsl_sf_bessel_jl_steed_array(r_grid->n[iL], k*r, &(tdsfm->jl[(r_grid->n[iL]+1)*ik]));
	}

	tdsfm->ylm = ylm_cache_new(r_grid->n[iL], k_grid);

	tdsfm->int_A  = 0.0;
	tdsfm->int_A2 = 0.0;

	return tdsfm;
}

void tdsfm_del(tdsfm_t* tdsfm) {
	ylm_cache_del(tdsfm->ylm);
	free(tdsfm->jl);
	free(tdsfm->data);
	free(tdsfm);
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

	double Az  = t*At - tdsfm->int_A;
	double A_2 = t*pow(At, 2) - 2*At*tdsfm->int_A + tdsfm->int_A2;

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<Nk; ik++) {
		for (int ic=0; ic<tdsfm->k_grid->n[iC]; ic++) {
			double k = sp_grid_r(tdsfm->k_grid, ik);
			double kz = sp_grid_c(tdsfm->k_grid, ic)*k;

			for (int il=0; il<wf->grid->n[iL]; il++) {
				tdsfm->data[ik+ic*Nk] += r/sqrt(2*M_PI)*cpow(-I, il+1)*cexp(0.5*I*(k*k*t + 2*kz*Az + A_2))*(tdsfm->jl[il+ik*(Nl+1)]*((swf_get(wf, ir+1, il) - swf_get(wf, ir-1, il))/(2*dr)-(il+1)*swf_get(wf, ir, il)/r) + k*swf_get(wf, ir, il)*tdsfm->jl[il+1+ik*(Nl+1)])*ylm_cache_get(tdsfm->ylm, il, wf->m, ic);
			}
		}
	}
}