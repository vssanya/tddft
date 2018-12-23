from wavefunc cimport ShWavefunc, sh_wavefunc_from_point


cdef class ShWavefuncArrayGPU:
    def __cinit__(self, ShWavefunc wf, int N):
        self.grid = wf.grid
        self.cdata = new cShWavefuncArrayGPU(wf.cdata[0], N)

    def __init__(self, ShWavefunc wf, int N):
        pass

    def __dealloc__(self):
        del self.cdata

    def get(self, int i):
        return sh_wavefunc_from_point(self.cdata.get(i), self.grid, True)

    @property
    def N(self):
        return self.cdata.N


cdef class ShWavefuncGPU:
    def __cinit__(self, ShWavefunc wf):
        self.grid = wf.grid
        self.cdata = new cShWavefuncGPU(wf.cdata[0])

    def __init__(self, ShWavefunc wf):
        pass

    def __dealloc__(self):
        del self.cdata

    def get(self):
        return sh_wavefunc_from_point(self.cdata.get(), self.grid, True)
