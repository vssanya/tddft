from grid cimport cGrid2d

cdef extern from "array.h":
    cdef cppclass Array2D[T]:
        Array2D()
        Array2D(cGrid2d& grid)
        Array2D(T* data, cGrid2d& grid)

        T* data

cdef class DoubleArray2D:
    cdef Array2D[double]* cdata
