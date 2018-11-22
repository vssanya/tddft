#pragma once

#include "../sh_wavefunc_gpu.h"
#include "../atom.h"
#include "../abs_pot.h"
#include "../fields.h"


namespace workspace {
	class WfGpu {
		public:
			WfGpu(AtomCache const& atomCache, ShGrid const& grid, UabsCache const& uabsCache, int gpuGridNl = 1024, int threadsPerBlock=32);

			~WfGpu();

			void prop(ShWavefuncGPU& wf, field_t const& field, double t, double dt);
			void prop_abs(ShWavefuncGPU& wf, double dt);
			void prop_at(ShWavefuncGPU& wf, double dt);

			ShGrid const& grid;

			int d_gridNl;
			int threadsPerBlock;

			UabsCache const& uabsCache;
			AtomCache const& atomCache;

            cdouble* d_alpha;
            cdouble* d_betta;

            double* d_uabs;
			double* d_atomU;
	};
};
