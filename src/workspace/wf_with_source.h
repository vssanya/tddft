#pragma once

#include "wf.h"


namespace workspace {
    class WfEWithSource: public WfBase {
		public:
            WfEWithSource(
					ShGrid    const& grid,
					AtomCache<ShGrid> const* atom_cache,
					UabsCache const& uabs,
					ShWavefunc const& wf_source,
					double E,
					PropAtType propAtType,
					Gauge gauge,
					int num_threads
					):
                WfBase(grid, atom_cache, uabs, propAtType, gauge, num_threads), wf_source(wf_source), source_E(E), abs_norm(0.0)
		{
			assert(grid.n[iR] == wf_source.grid.n[iR]);
		}

			void prop_abs(ShWavefunc& wf, double dt);
			void prop_src(ShWavefunc& wf, field_t const* field, double t, double dt);
            void prop(ShWavefunc& wf, field_t const* field, double t, double dt) {
                WfBase::prop(wf, field, t, dt);
				prop_src(wf, field, t, dt);
			}

		private:
			ShWavefunc const& wf_source;
			double source_E;

		public:
			double abs_norm;
	};
}
