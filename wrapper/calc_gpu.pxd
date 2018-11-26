from wavefunc_gpu cimport cShWavefuncArrayGPU, cShWavefuncGPU
from atom cimport cAtomCache
from field cimport field_t


cdef extern from "calc_gpu.h":
    double* calc_wf_array_gpu_az(cShWavefuncArrayGPU& wf_array, cAtomCache &atom_cache, double* E, double* res);

    double calc_wf_gpu_az(
            cShWavefuncGPU& wf,
            cAtomCache& atom_cache,
            field_t* field,
            double t
    )
