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

class TDSFM_Base {
	public:
		sp_grid_t const* k_grid;
		sh_grid_t const* r_grid;

		int ir;

		cdouble* data; //!< data[i + j*grid->Np] = \f$a_{I}(k_{i,j}\f$

		sp_grid_t const* jl_grid;
		jl_cache_t* jl;

		sp_grid_t const* ylm_grid;
		ylm_cache_t* ylm;

		double int_A;
		double int_A2;

		TDSFM_Base(sp_grid_t const* k_grid, sh_grid_t const* r_grid, int ir, sp_grid_t const* jl_grid, sp_grid_t const* ylm_grid);
		virtual ~TDSFM_Base();

		void init_cache();

		virtual void calc(field_t const* field, sh_wavefunc_t const& wf, double t, double dt) = 0;
		virtual void calc_inner(field_t const* field, sh_wavefunc_t const& wf, double t, int ir_min, int ir_max) = 0;

		double pz() const;

		inline
			cdouble operator()(int ik, int ic) const {
				return data[ik + ic*k_grid->n[iR]];
			}

		inline
			cdouble& operator()(int ik, int ic) {
				return data[ik + ic*k_grid->n[iR]];
			}
};

struct TDSFM_E: public TDSFM_Base {
	TDSFM_E(sp_grid_t const* k_grid, sh_grid_t const* r_grid, double A_max, int ir, bool init_cache=true);
	~TDSFM_E();

	void calc(field_t const* field, sh_wavefunc_t const& wf, double t, double dt);
	void calc_inner(field_t const* field, sh_wavefunc_t const& wf, double t, int ir_min, int ir_max);
};


struct TDSFM_A: public TDSFM_Base {
	TDSFM_A(sp_grid_t const* k_grid, sh_grid_t const* r_grid, int ir, bool init_cache=true);
	~TDSFM_A();

	void calc(field_t const* field, sh_wavefunc_t const& wf, double t, double dt);
	void calc_inner(field_t const* field, sh_wavefunc_t const& wf, double t, int ir_min, int ir_max);
};
