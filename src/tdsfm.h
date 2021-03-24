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
#include "wavefunc/sh_2d.h"
#include "sphere_harmonics.h"

class TDSFM_Base {
	public:
		SpGrid const k_grid;
		ShGrid const r_grid;

		int ir;
		int m_max;

		cdouble* data; //!< data[i + j*grid->Np] = \f$a_{I}(k_{i,j}\f$

		SpGrid jl_grid;
		JlCache* jl;

		SpGrid ylm_grid;
		YlmCache* ylm;

		double int_A;
		double int_A2;

		TDSFM_Base(SpGrid const k_grid, ShGrid const r_grid, int ir, int m_max, bool own_data = true);
		virtual ~TDSFM_Base();

		void init_cache();

        virtual void calc(
				field_t const* field,
				ShWavefunc const& wf,
				double t, double dt,
				double mask = 1.0,
				cdouble* data = nullptr) = 0;

        virtual void calc_inner(
				field_t const* field, 
				ShWavefunc const& wf, 
				double t, 
				int ir_min, int ir_max, 
				int l_min = 0, int l_max = -1,
				cdouble* data = nullptr) = 0;

		double pz() const;
		double norm() const;

		void calc_norm_k(ShWavefunc const& wf, int ir_min, int ir_max, int l_min, int l_max, cdouble* data = nullptr);

		inline
			cdouble operator()(int ik, int ic) const {
				return data[ik + ic*k_grid.n[iR]];
			}

		inline
			cdouble& operator()(int ik, int ic) {
				return data[ik + ic*k_grid.n[iR]];
			}
};

class TDSFM_E: public TDSFM_Base {
	public:
		TDSFM_E(SpGrid const k_grid, ShGrid const r_grid, double A_max, int ir, int m_max, bool init_cache=true, bool own_data=true);
		~TDSFM_E();

		void calc(field_t const* field, ShWavefunc const& wf, double t, double dt, double mask, cdouble* data = nullptr);
		void calc_inner(field_t const* field, ShWavefunc const& wf, double t, int ir_min, int ir_max, int l_min = 0, int l_max = -1, cdouble* data = nullptr);
};


class TDSFM_A: public TDSFM_Base {
	public:
		TDSFM_A(SpGrid const k_grid, ShGrid const r_grid, int ir, int m_max, bool init_cache=true, bool own_data=true);
		~TDSFM_A();

		void calc(field_t const* field, ShWavefunc const& wf, double t, double dt, double mask, cdouble* data = nullptr);
		void calc_inner(field_t const* field, ShWavefunc const& wf, double t, int ir_min, int ir_max, int l_min = 0, int l_max = -1, cdouble* data = nullptr);
};


#include "workspace/wf.h"
#include "orbitals.h"

class TDSFMOrbs {
	public:
		TDSFMOrbs(Orbitals<ShGrid> const& orbs, SpGrid const k_grid, int ir, workspace::Gauge gauge, bool init_cache=true, double A_max = 0.0);
		~TDSFMOrbs();

		TDSFM_Base* tdsfmWf;
		Orbitals<SpGrid2d>* pOrbs;
		SpGrid2d grid;

		void calc(field_t const* field, Orbitals<ShGrid> const& orbs, double t, double dt, double mask);
		void calc_inner(field_t const* field, Orbitals<ShGrid> const& orbs, double t, int ir_min, int ir_max, int l_min = 0, int l_max = -1);

		void collect(cdouble* dest) const;
};
