from wavefunc_gpu cimport ShWavefuncGPU, ShWavefuncArrayGPU
from field cimport Field
from atom cimport ShAtomCache


cdef class WfArrayGPUWorkspace:
    def __cinit__(self, ShAtomCache atom_cache, ShGrid grid, UabsCache uabs_cache, int N):
        self.cdata = new WfArrayGpu(atom_cache.cdata, grid.data, uabs_cache.cdata, N)

        self.atom_cache = atom_cache
        self.grid = grid
        self.uabs_cache = uabs_cache

    def __init__(self, ShAtomCache atom_cache, ShGrid grid, UabsCache uabs_cache, int N):
        pass

    def __dealloc__(self):
        del self.cdata

    def prop(self, ShWavefuncArrayGPU wf_array, double[:] E, double dt):
        self.cdata.prop(wf_array.cdata, &E[0], dt)

    def prop_abs(self, ShWavefuncArrayGPU wf_array, double dt):
        self.cdata.prop_abs(wf_array.cdata,  dt)

    def prop_at(self, ShWavefuncArrayGPU wf_array, double dt):
        self.cdata.prop_at(wf_array.cdata,  dt)


cdef class WfGPUWorkspace:
    def __cinit__(self, ShAtomCache atom_cache, ShGrid grid, UabsCache uabs_cache, int gpuGridNl = 1024, int threadsPerBlock = 32):
        self.cdata = new WfGpu(atom_cache.cdata[0], grid.data[0], uabs_cache.cdata[0], gpuGridNl, threadsPerBlock)

        self.atom_cache = atom_cache
        self.grid = grid
        self.uabs_cache = uabs_cache

    def __init__(self, ShAtomCache atom_cache, ShGrid grid, UabsCache uabs_cache, int gpuGridNl = 1024, int threadsPerBlock = 32):
        pass

    def __dealloc__(self):
        del self.cdata

    def prop(self, ShWavefuncGPU wf, Field field, double t, double dt):
        self.cdata.prop(wf.cdata[0], field.cdata[0], t, dt)

    def prop_abs(self, ShWavefuncGPU wf, double dt):
        self.cdata.prop_abs(wf.cdata[0],  dt)

    def prop_at(self, ShWavefuncGPU wf, double dt):
        self.cdata.prop_at(wf.cdata[0],  dt)
