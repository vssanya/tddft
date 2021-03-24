#pragma once

#include "../atom.h"
#include "../wavefunc/sh_arr.h"
#include "../abs_pot.h"

#include "wf.h"


namespace workspace {
	template <typename Grid>
	class WfArray {
		public:
			WfArray(
					Grid    const& grid,
					AtomCache<Grid> const* atom_cache,
					UabsCache const& uabs,
					PropAtType propAtType,
					Gauge gauge,
					int num_threads
					);

			~WfArray() {};

			void prop(WavefuncArray<Grid>* arr, double E[], double dt);

			workspace::WavefuncWS<Grid> ws;
	};
};
