#pragma once

#include "wf.h"


namespace workspace {
	class wf_E_with_source: public wf_E {
		public:
			wf_E_with_source(sh_grid_t const* grid, uabs_sh_t const* uabs, sh_wavefunc_t const& wf_source, int num_threads): wf_E(grid, uabs, num_threads), wf_source(wf_source)
		{
			assert(grid->n[iR] == wf_source.grid->n[iR]);
		}

			void prop_src(sh_wavefunc_t& wf, field_t const* field, double t, double dt);
			void prop(sh_wavefunc_t& wf, atom_t const* atom, field_t const* field, double t, double dt) {
				wf_E::prop(wf, atom, field, t, dt);
				prop_src(wf, field, t, dt);
			}

		private:
			sh_wavefunc_t const& wf_source;
	};
}
