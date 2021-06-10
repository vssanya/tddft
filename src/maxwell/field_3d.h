#pragma once

#include "mpi_utils.h"

#include "../array.h"

class Field3D {
public:
  Field3D(Grid3d const &grid, MPI_Comm mpi_comm);

  Grid3d grid;

  Array3D<double> Ex;
  Array3D<double> Ey;
  Array3D<double> Ez;

  Array3D<double> Hx;
  Array3D<double> Hy;
  Array3D<double> Hz;

#ifdef _MPI
  MPI_Comm mpi_comm;
#endif
};
