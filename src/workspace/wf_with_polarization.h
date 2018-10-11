#pragma once

#include "wf.h"


namespace workspace {
	class WfWithPolarization: public WfE {
		public:
			WfWithPolarization(AtomCache const* atom_cache, ShGrid const* grid, UabsCache const* uabs, double const* Upol_1, double const* Upol_2, int num_threads):
				WfE(atom_cache, grid, uabs, num_threads), Upol_1(Upol_1), Upol_2(Upol_2) {
				}

			void prop(ShWavefunc& wf, field_t const* field, double t, double dt);

		private:
			double const* Upol_1;
			double const* Upol_2;
	};
}
