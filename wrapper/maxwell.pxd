from grid cimport cGrid1d, Grid1d, cGrid3d, Grid3d
from carray cimport Array1D, Array3D

cdef extern from "const.h":
    double C_au

cdef extern from "maxwell/1d.h":
    cdef cppclass cWorkspace1D "maxwell::Workspace1D":
        cWorkspace1D(cGrid1d& grid);

        void prop(double dt);
        void prop(double dt, Array1D[double] eps);
        void prop(double dt, double* eps);
        void prop_pol(double dt, double* P);

        int move_center_window_to_max_E();

        cGrid1d& grid;

        Array1D[double] E;
        Array1D[double] D;
        Array1D[double] H;

cdef class MaxwellWorkspace1D:
    cdef:
        Grid1d grid
        cWorkspace1D* cdata

cdef extern from "maxwell/3d.h":
    cdef cppclass cWorkspace3D "maxwell::Workspace3D":
        cWorkspace3D(cGrid3d& grid);

        void prop(double dt);
        void prop(double dt, Array3D[double]* j);

        cGrid3d& grid;

        Array3D[double] Ex
        Array3D[double] Ey
        Array3D[double] Ez

        Array3D[double] Hx
        Array3D[double] Hy
        Array3D[double] Hz

cdef class MaxwellWorkspace3D:
    cdef:
        Grid3d grid
        cWorkspace3D* cdata
