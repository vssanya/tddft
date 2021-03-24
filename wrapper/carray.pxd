from grid cimport cGrid3d, cGrid2d, Grid2d, cGrid1d

cdef extern from "array.h":
    cdef cppclass Array3D[T]:
        Array3D()
        Array3D(cGrid3d& grid)
        Array3D(T* data, cGrid3d& grid)

        T* data

    cdef cppclass Array2D[T]:
        Array2D()
        Array2D(cGrid2d& grid)
        Array2D(T* data, cGrid2d& grid)

        T* data

    cdef cppclass Array1D[T]:
        Array1D()
        Array1D(cGrid1d& grid)
        Array1D(T* data, cGrid1d& grid)

        T* data

cdef class DoubleArray2D:
    cdef Array2D[double]* cdata
    cdef Grid2d grid
