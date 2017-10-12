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

	double func_A(int it)   { return     field_A(field, t) - field_A(field, it*dt); }
	double func_A_2(int it) { return pow(field_A(field, t) - field_A(field, it*dt), 2); }

	double Az = integrate_1d(func_A, (int)(t/dt+1), dt);
	double A_2 = 0.0;

	for (int ik=0; ik<Nk; ik++) {
		double k = sp_grid_r(tdsfm->k_grid, ik);
		for (int ic=0; ic<tdsfm->k_grid->n[iC]; ic++) {
			double kz = sp_grid_c(tdsfm->k_grid, ic);

			for (int il=0; il<wf->grid->n[iL]; il++) {
				tdsfm->data[ik+ic*Nk] += r/sqrt(2*M_PI)*pow(-I, il+1)*exp(0.5*I*(k*k*t + 2*kz*A_z + A_2))*(tdsfm->jl[il+ik*(Nl+1)]*((swf_get(wf, ir+1, il) - swf_get(wf, ir-1, il))/(2*dr)-(il+1)*swf_get(wf, ir, il)/r) + k*swf_get(wf, ir, il)*tdsfm->jl[il+1+ik*(Nl+1)])*ylm_cache_get(tdsfm->ylm, il, wf->m, ic);
			}
		}
	}
}
