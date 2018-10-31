from wavefunc_gpu cimport cShWavefuncArrayGPU
from atom cimport cAtomCache

cdef extern from "calc_gpu.h":
    double* calc_wf_array_gpu_az(cShWavefuncArrayGPU& wf_array, cAtomCache &atom_cache, double* E, double* res);
