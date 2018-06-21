#pragma once

#include "wf.h"


namespace workspace {
    class WfEWithSource: public WfE {
		public:
            WfEWithSource(AtomCache const* atom_cache, ShGrid const* grid, UabsCache const* uabs, ShWavefunc const& wf_source, double E, int num_threads):
                WfE(atom_cache, grid, uabs, num_threads), wf_source(wf_source), source_E(E), abs_norm(0.0)
		{
			assert(grid->n[iR] == wf_source.grid->n[iR]);
		}

			void prop_abs(ShWavefunc& wf, double dt);
			void prop_src(ShWavefunc& wf, field_t const* field, double t, double dt);
            void prop(ShWavefunc& wf, field_t const* field, double t, double dt) {
                WfE::prop(wf, field, t, dt);
				prop_src(wf, field, t, dt);
			}

			double abs_norm;

		private:
			ShWavefunc const& wf_source;
			double source_E;
	};
}
