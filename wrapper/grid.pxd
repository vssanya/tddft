cdef extern from "grid.h":
    ctypedef struct grid2_t:
        int    n[2]
        double d[2]

    ctypedef struct grid3_t:
        int    n[3]
        double d[3]

    ctypedef grid2_t sh_grid_t
    ctypedef grid3_t sp_grid_t

    size_t grid2_size(grid2_t* grid)
    size_t grid3_size(grid3_t* grid)

    size_t grid2_index(grid2_t* grid, int i[2])
    size_t grid3_index(grid3_t* grid, int i[3])

    sp_grid_t* sp_grid_new(int n[3], double r_max)
    sh_grid_t* sh_grid_new(int n[2], double r_max)
    void sh_grid_del(sh_grid_t* grid)
    void sp_grid_del(sp_grid_t* grid)

    double sp_grid_r(sp_grid_t* grid, int ir)
    double sp_grid_c(sp_grid_t* grid, int ic)
    double sp_grid_phi(sp_grid_t* grid, int ip)
    double sh_grid_r(sh_grid_t* grid, int ir)
    double sh_grid_r_max(sh_grid_t* grid)
    int sh_grid_l(sh_grid_t* grid, int il)
    int sh_grid_m(sh_grid_t* grid, int im)

    grid2_t* ct_grid_new(int n[2], double x_max, double y_max);
    double ct_grid_x(grid2_t* grid, int ix);
    double ct_grid_y(grid2_t* grid, int iy);

    grid2_t* sp2_grid_new(int n[2], double r_max);
    double sp2_grid_r(grid2_t* grid, int ir);
    double sp2_grid_c(grid2_t* grid, int ic);

cdef class CtGrid:
    cdef grid2_t* cdata

cdef class Sp2Grid:
    cdef grid2_t* cdata

cdef class ShGrid:
    cdef sh_grid_t* data

cdef class SpGrid:
    cdef sp_grid_t* data
