from types cimport cdouble
from wavefunc cimport cShWavefunc
from atom cimport cAtomCache
from grid cimport cShGrid, ShGrid

cdef extern from "sh_wavefunc_gpu.h":
    cdef cppclass cShWavefuncArrayGPU "ShWavefuncArrayGPU":
        cShGrid* grid
        int N

        cdouble* data
        bint data_own

        int m

        cShWavefuncArrayGPU(cShWavefunc& wf, int N)
        cShWavefuncArrayGPU(cdouble* data, cShGrid* grid, int m, int N)
        cShWavefuncArrayGPU(cShGrid* grid, int m, int N)

        cShWavefunc* get(int i)

    cdef cppclass cShWavefuncGPU "ShWavefuncGPU":
        cShGrid* grid
        cdouble* data
        bint data_own

        int m

        cShWavefuncGPU(cdouble* data, cShGrid* grid, int m)
        cShWavefuncGPU(cShWavefunc& wf)
        cShWavefuncGPU(cShGrid* grid, int m)

        cShWavefunc* get(int i)


cdef class ShWavefuncArrayGPU:
    cdef cShWavefuncArrayGPU* cdata
    cdef bint dealloc
    cdef public ShGrid grid
