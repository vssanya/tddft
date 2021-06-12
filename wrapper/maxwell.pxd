from grid cimport cGrid1d, Grid1d, cGrid3d, Grid3d
from carray cimport Array1D, Array3D

from mpi4py.libmpi cimport MPI_Comm

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

cdef extern from "maxwell/field_3d.h":
    cdef cppclass cField3D "Field3D":
        cField3D(cGrid3d &grid, MPI_Comm mpi_comm);

        cGrid3d& grid;

        Array3D[double] Ex
        Array3D[double] Ey
        Array3D[double] Ez

        Array3D[double] Dx
        Array3D[double] Dy
        Array3D[double] Dz

        Array3D[double] Hx
        Array3D[double] Hy
        Array3D[double] Hz

        Array3D[double] Bx
        Array3D[double] By
        Array3D[double] Bz

        double E()

cdef class Field3D:
    cdef:
        Grid3d grid
        cField3D* cdata

cdef extern from "maxwell/3d.h":
    cdef cppclass cWorkspace3D "maxwell::Workspace3D":
        cWorkspace3D(cGrid3d& grid, MPI_Comm mpi_comm);

        void prop(cField3D field, double dt);
        void prop(cField3D field, double dt, Array3D[double]* j);
        void prop(cField3D field, double dt, Array3D[double] sigma);
        void prop(cField3D field, double dt, Array3D[double] sigma, double labs, double sigma_abs);

        void get_sigma_abs(double labs, double* res);

        cGrid3d& grid;

cdef class MaxwellWorkspace3D:
    cdef:
        Grid3d grid
        cWorkspace3D* cdata

cdef extern from "maxwell/1d.h":
    cdef cppclass cWorkspaceCyl1D "maxwell::WorkspaceCyl1D":
        cWorkspaceCyl1D(cGrid1d& grid);

        void prop(double dt);
        void prop(double dt, double* N, double nu);

        cGrid1d& grid;

        Array1D[double] Er;
        Array1D[double] Ephi;
        Array1D[double] Hz;

        Array1D[double] jr;
        Array1D[double] jphi;

cdef class MaxwellWorkspaceCyl1D:
    cdef:
        Grid1d grid
        cWorkspaceCyl1D* cdata
