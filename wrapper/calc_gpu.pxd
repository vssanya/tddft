from wavefunc_gpu cimport cShWavefuncArrayGPU, cShWavefuncGPU
from atom cimport AtomCache
from field cimport field_t
from grid cimport cShGrid


cdef extern from "calc_gpu.h":
    double* calc_wf_array_gpu_az(cShWavefuncArrayGPU& wf_array, AtomCache[cShGrid] &atom_cache, double* E, double* res);

    double calc_wf_gpu_az(
            cShWavefuncGPU& wf,
            AtomCache[cShGrid]& atom_cache,
            field_t* field,
            double t
    )
