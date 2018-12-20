cdef extern from "grid.h":
    cdef cppclass cGrid1d "Grid1d":
        int n
        double d

        cGrid1d()
        cGrid1d(int n, double d)

    cdef cppclass cGrid2d "Grid2d":
        int n[2]
        double d[2]

    cdef cppclass cGrid3d "Grid3d":
        int n[3]
        double d[3]

    cdef cppclass cShGrid "ShGrid":
        cShGrid(int n[2], double Rmax)

        double r(int ir)
        double Rmax()
        int l(int il)
        int m(int im)

        int n[2]
        double d[2]

    cdef cppclass cShNeGrid "ShNotEqudistantGrid":
        cShNeGrid(double Rmin, double Rmax, double Ra, double dr_max, int Nl)

        double r(int ir)
        double Rmax()
        int l(int il)
        int m(int im)

        int n[2]
        double d[2]

    cdef cppclass cSpGrid "SpGrid":
        cSpGrid(int n[3], double Rmax)
        double r(int ir)
        double Rmax()
        double c(int ic)
        double phi(int ip)
        int ir(double r)
        int ic(double c)

        int n[3]
        double d[3]

    cdef cppclass cSpGrid2d "SpGrid2d":
        cSpGrid2d(int n[2], double Rmax)
        double r(int ir)
        double c(int ic)

    cdef cppclass cCtGrid "CtGrid":
        cCtGrid(int n[2], double x_max, double y_max)
        double x(int ix)
        double y(int iy)

cdef class Grid1d:
    cdef cGrid1d cdata

cdef class CtGrid:
    cdef cCtGrid* cdata

cdef class SpGrid2d:
    cdef cSpGrid2d* cdata

cdef class ShGrid:
    cdef cShGrid* data

cdef class ShNeGrid:
    cdef cShNeGrid* data
    cdef double Rmin
    cdef double Ra

cdef class SpGrid:
    cdef cSpGrid* data
