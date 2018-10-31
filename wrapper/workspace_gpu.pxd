from atom cimport cAtomCache
from abs_pot cimport cUabsCache
from wavefunc_gpu cimport cShWavefuncArrayGPU


cdef extern from "workspace/wf_array_gpu.h" namespace "workspace":
    cdef cppclass WfArrayGpu:
    WfArrayGpu(cAtomCache* atomCache, cShGrid* grid, cUabsCache* uabsCache, int N);

    void prop(cShWavefuncArrayGPU* wf, double* E, double dt);
    void prop_abs(cShWavefuncArrayGPU* wf, double dt);
    void prop_at(cShWavefuncArrayGPU* wf, double dt);
