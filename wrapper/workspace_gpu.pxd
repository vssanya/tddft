from atom cimport cAtomCache, AtomCache
from abs_pot cimport cUabsCache, UabsCache
from grid cimport cShGrid, ShGrid
from wavefunc_gpu cimport cShWavefuncArrayGPU, cShWavefuncGPU
from field cimport field_t


cdef extern from "workspace/wf_gpu.h" namespace "workspace":
    cdef cppclass WfGpu:
        WfGpu(cAtomCache& atomCache, cShGrid& grid, cUabsCache& uabsCache, int gpuGridNl, int threadsPerBlock)
        void prop(cShWavefuncGPU& wf, field_t& field, double t, double dt)
        void prop_abs(cShWavefuncGPU& wf, double dt)
        void prop_at(cShWavefuncGPU& wf, double dt)


cdef extern from "workspace/wf_array_gpu.h" namespace "workspace":
    cdef cppclass WfArrayGpu:
        WfArrayGpu(cAtomCache* atomCache, cShGrid* grid, cUabsCache* uabsCache, int N)

        void prop(cShWavefuncArrayGPU* wf, double* E, double dt)
        void prop_abs(cShWavefuncArrayGPU* wf, double dt)
        void prop_at(cShWavefuncArrayGPU* wf, double dt)


cdef class WfArrayGPUWorkspace:
    cdef:
        WfArrayGpu* cdata
        AtomCache atom_cache
        ShGrid grid
        UabsCache uabs_cache


cdef class WfGPUWorkspace:
    cdef:
        WfGpu* cdata
        AtomCache atom_cache
        ShGrid grid
        UabsCache uabs_cache
