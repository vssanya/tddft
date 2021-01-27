from grid cimport cGrid1d, Grid1d
from carray cimport Array1D


cdef extern from "maxwell/1d.h":
    cdef cppclass cWorkspace1D "maxwell::Workspace1D":
        cWorkspace1D(cGrid1d& grid);

        void prop(double dt);
        void prop(double dt, Array1D[double] eps);
        void prop(double dt, double* eps);

        cGrid1d& grid;

        Array1D[double] E;
        Array1D[double] D;
        Array1D[double] H;

cdef class MaxwellWorkspace1D:
    cdef:
        Grid1d grid
        cWorkspace1D* cdata
