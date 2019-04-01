#pragma once


#include "../sh_wavefunc_gpu.h"
#include "../atom.h"
#include "../abs_pot.h"

namespace workspace {
	class WfArrayGpu {
		public:
			WfArrayGpu(AtomCache<ShGrid> const* atomCache, ShGrid const* grid, UabsCache const* uabsCache, int N);

			~WfArrayGpu();

			void prop(ShWavefuncArrayGPU* wf, double E[], double dt);
			void prop_abs(ShWavefuncArrayGPU* wf, double dt);
			void prop_at(ShWavefuncArrayGPU* wf, double dt);

			ShGrid const* grid;
			int N;

			UabsCache const* uabsCache;
			AtomCache<ShGrid> const* atomCache;

            cdouble* d_alpha;
            cdouble* d_betta;

            double* d_uabs;
			double* d_atomU;
			double* d_E;
	};
};
