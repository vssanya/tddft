cdef extern from "grid.h":
    cdef cppclass cRange "ShGrid::RangeR":
        int r_min
        int r_max

        cRange(int r_min, int r_max)
        cRange()

    cdef cppclass cGrid1d "Grid1d":
        int n
        double d

        cGrid1d()
        cGrid1d(int n, double d)

        size_t size()

    cdef cppclass cGrid2d "Grid2d":
        cGrid2d()
        cGrid2d(int nx, int ny)

        int n[2]
        double d[2]

        size_t size()

    cdef cppclass cGrid3d "Grid3d":
        int n[3]
        double d[3]

        size_t size()

        cGrid3d()
        cGrid3d(int nx, int ny, int nz, double dx, double dy, double dz)

    cdef cppclass cShGrid "ShGrid":
        cShGrid(int n[2], double Rmax)

        double r(int ir)
        double Rmax()
        int l(int il)
        int m(int im)

        cRange getRange(double rmax)

        int n[2]
        double d[2]

    cdef cppclass cShNeGrid "ShNotEqudistantGrid":
        cShNeGrid(double Rmin, double Rmax, double Ra, double dr_max, int Nl)

        double r(int ir)
        double Rmax()
        int l(int il)
        int m(int im)

        cRange getRange(double rmax)

        int n[2]
        double d[2]

    cdef cppclass cShNeGrid3D "ShNotEqudistantGrid3D":
        cShNeGrid3D(double Rmin, double Rmax, double Ra, double dr_max, int Nl)

        double r(int ir)
        double Rmax()
        int l(int il)
        int m(int im)

        cRange getRange(double rmax)

        int n[2]
        double d[2]

    cdef cppclass cSpGrid2d "SpGrid2d":
        cSpGrid2d(int n[2], double Rmax)
        double r(int ir)
        double c(int ic)

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

        cSpGrid2d getGrid2d()

    cdef cppclass cCtGrid "CtGrid":
        cCtGrid(int n[2], double x_max, double y_max)
        double x(int ix)
        double y(int iy)

cdef class Range:
    cdef cRange cdata

cdef class Grid1d:
    cdef cGrid1d cdata

cdef class Grid2d:
    cdef cGrid2d cdata

cdef class Grid3d:
    cdef cGrid3d cdata

cdef class CtGrid:
    cdef cCtGrid* cdata

cdef class SpGrid2d:
    cdef cSpGrid2d* data

cdef class ShGrid:
    cdef cShGrid* data

cdef class ShNeGrid:
    cdef cShNeGrid* data
    cdef double Rmin
    cdef double Ra

cdef class ShNeGrid3D:
    cdef cShNeGrid3D* data
    cdef double Rmin
    cdef double Ra

cdef class SpGrid:
    cdef cSpGrid* data
