#include "calc_gpu.h"

double* calc_wf_array_gpu_az(ShWavefuncArrayGPU const &wf_array, const AtomCache &atom_cache, double E[], double* res) {
	res = wf_array.cos(atom_cache.data_dudz, res);

	for (int in = 0; in < wf_array.N; in++) {
		res[in] = - E[in] - res[in];
	}

	return res;
}

double calc_wf_gpu_az(
		ShWavefuncGPU const& wf,
		const AtomCache& atom_cache,
		field_t const* field,
		double t
) {
	return - field_E(field, t) - wf.cos(atom_cache.data_dudz);
}
