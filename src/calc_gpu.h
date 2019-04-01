#pragma once

#include "fields.h"

#include "sh_wavefunc_gpu.h"
#include "atom.h"


double* calc_wf_array_gpu_az(
		ShWavefuncArrayGPU const& wf_array,
		AtomCache<ShGrid>& atom_cache,
		double E[],
		double* res = nullptr
);

double calc_wf_gpu_az(
		ShWavefuncGPU const& wf,
		AtomCache<ShGrid>& atom_cache,
		field_t const* field,
		double t
);
