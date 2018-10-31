#pragma once

#include "sh_wavefunc_gpu.h"
#include "atom.h"


double* calc_wf_array_gpu_az(ShWavefuncArrayGPU const& wf_array, const AtomCache &atom_cache, double E[], double* res = nullptr);
