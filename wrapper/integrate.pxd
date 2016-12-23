cdef extern from "integrate.h":
    ctypedef double (*func_1d_t)(int i)
    ctypedef double (*func_2d_t)(int ix, int iy)
    ctypedef double (*func_3d_t)(int ix, int iy, int iz)
    double integrate_1d(func_1d_t f, int nx, double dx)
