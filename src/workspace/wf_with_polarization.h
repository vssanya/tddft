#pragma once

#include "wf.h"


namespace workspace {
	class WfWithPolarization: public WfBase {
		public:
			WfWithPolarization(
					ShGrid    const& grid,
					AtomCache<ShGrid> const& atom_cache,
					UabsCache const& uabs,
					double const* Upol_1,
					double const* Upol_2,
					PropAtType propAtType,
					Gauge gauge,
					int num_threads
					):
				WfBase(grid, atom_cache, uabs, propAtType, gauge, num_threads),
				Upol_1(Upol_1), Upol_2(Upol_2)
		{}

			void prop(ShWavefunc& wf, field_t const* field, double t, double dt);

		private:
			double const* Upol_1;
			double const* Upol_2;
	};
}
