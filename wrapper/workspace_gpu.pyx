from wavefunc_gpu cimport ShWavefuncArrayGPU


cdef class WfArrayGPUWorkspace:
    def __cinit__(self, AtomCache atom_cache, ShGrid grid, UabsCache uabs_cache, int N):
        self.atom_cache = atom_cache
        self.grid = grid
        self.uabs_cache = uabs_cache

        self.cdata = new WfArrayGpu(atom_cache.cdata, grid.data, uabs_cache.cdata, N)

    def __init__(self, AtomCache atom_cache, ShGrid grid, UabsCache uabs_cache, int N):
        pass

    def prop(self, ShWavefuncArrayGPU wf_array, double[:] E, double dt):
        self.cdata.prop(wf_array.cdata, &E[0], dt)
