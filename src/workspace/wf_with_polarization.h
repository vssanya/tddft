#pragma once

#include "wf.h"


namespace workspace {
	class WfWithPolarization: public WfE {
		public:
			WfWithPolarization(AtomCache const* atom_cache, ShGrid const* grid, UabsCache const* uabs, double const* Upol, int num_threads):
				WfE(atom_cache, grid, uabs, num_threads), Upol(Upol) {
				}

			void prop(ShWavefunc& wf, field_t const* field, double t, double dt);

		private:
			double const* Upol;
	};
}
