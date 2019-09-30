from grid cimport cSpGrid2d

cdef extern from "array.h":
    cdef cppclass ArraySp2D[T]: 
        ArraySp2D()
        ArraySp2D(T* data, cSpGrid2d& grid)
