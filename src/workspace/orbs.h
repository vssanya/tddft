#pragma once

#include "wf.h"
#include "../orbitals.h"

#include "../hartree_potential.h"
#include "../utils.h"
#include "../types.h"

namespace workspace
{
	class orbs {
		public:
			orbs(sh_grid_t const* sh_grid, sp_grid_t const* sp_grid, uabs_sh_t const* uabs, ylm_cache_t const* ylm_cache, int Uh_lmax, int Uxc_lmax, potential_xc_f Uxc, int num_threads);
			virtual ~orbs();

			void prop(orbitals_t* orbs, atom_t const* atom, field_t const* field, double t, double dt, bool calc_uee);
			void prop_img(orbitals_t* orbs, atom_t const* atom, double dt);
			void calc_Uee(orbitals_t const* orbs, int Uxc_lmax, int Uh_lmax);

			workspace::wf_base wf_ws;

			double* Utmp;
			double* Utmp_local;

			int Uh_lmax;

			potential_xc_f Uxc;
			int Uxc_lmax;

			double* Uee;
			int lmax;

			sh_grid_t const* sh_grid;
			sp_grid_t const* sp_grid;
			double* uh_tmp;
			double* n_sp; // for root
			double* n_sp_local;
			ylm_cache_t const* ylm_cache;
	};
} /* workspace */ 
