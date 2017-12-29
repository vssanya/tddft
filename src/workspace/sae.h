/*
 * =====================================================================================
 *
 *       Filename:  sae_workspace.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01.11.2017 16:41:16
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include "wf.h"
#include "../orbitals.h"

namespace workspace {
	class SAE {
		public:
			SAE(sh_grid_t const* grid, uabs_sh_t const* uabs, int num_threads);
			~SAE();

			void setGroundState(orbitals_t* gs_orbs) { this->gs_orbs = gs_orbs; }
			void prop(sh_wavefunc_t* wf, atom_t const* atom, field_t const* field, double t, double dt);

		private:
			workspace::wf_base* ws_wf;
			orbitals_t* gs_orbs;
	};
}
