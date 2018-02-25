#pragma once

#include "wf.h"


namespace workspace {
    class WfEWithSource: public WfE {
		public:
            WfEWithSource(Atom const& atom, ShGrid const* grid, uabs_sh_t const* uabs, ShWavefunc const& wf_source, double E, int num_threads): WfE(atom, grid, uabs, num_threads), wf_source(wf_source), source_E(E)
		{
			assert(grid->n[iR] == wf_source.grid->n[iR]);
		}

			void prop_src(ShWavefunc& wf, field_t const* field, double t, double dt);
            void prop(ShWavefunc& wf, field_t const* field, double t, double dt) {
                WfE::prop(wf, field, t, dt);
				prop_src(wf, field, t, dt);
			}

		private:
			ShWavefunc const& wf_source;
			double source_E;
	};
}
