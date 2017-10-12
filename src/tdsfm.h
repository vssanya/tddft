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

#include "types.h"
#include "grid.h"
#include "fields.h"
#include "sh_wavefunc.h"
#include "sphere_harmonics.h"


typedef struct {
	sp_grid_t const* k_grid;
	sh_grid_t const* r_grid;

	int ir;

	cdouble* data; //!< data[i + j*grid->Np] = \f$a_{I}(k_{i,j}\f$
	double* jl; // Precompute regular spherical Bessel function

	ylm_cache_t* ylm;
} tdsfm_t;
