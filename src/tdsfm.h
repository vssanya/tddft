/*
 * =====================================================================================
 *
 *       Filename:  tdsfm.h
 *
 *    Description:  Time-dependent surface flux method
 *
 *        Version:  1.0
 *        Created:  11.10.2017 15:00:59
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Romanov Alexander, 
 *   Organization:  IAP RAS
 *
 * =====================================================================================
 */

#pragma once

#include "types.h"
#include "grid.h"
#include "fields.h"
#include "sh_wavefunc.h"
#include "sphere_harmonics.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	sp_grid_t const* k_grid;
	sh_grid_t const* r_grid;

	int ir;

	cdouble* data; //!< data[i + j*grid->Np] = \f$a_{I}(k_{i,j}\f$

	sp_grid_t* appr_k_grid;
	double* jl; // Precompute regular spherical Bessel function
	ylm_cache_t* ylm;

	double int_A;
	double int_A2;
} tdsfm_t;

tdsfm_t* tdsfm_new(sp_grid_t const* k_grid, sh_grid_t const* r_grid, double A_max, int ir);
void tdsfm_del(tdsfm_t* tdsfm);
void tdsfm_calc(tdsfm_t* tdsfm, field_t const* field, sh_wavefunc_t const* wf, double t, double dt);
void tdsfm_calc_inner(tdsfm_t* tdsfm, field_t const* field, sh_wavefunc_t const* wf, double t, int ir_min, int ir_max);

double jn(int l, double x);

#ifdef __cplusplus
}
#endif
