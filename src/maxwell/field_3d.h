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

  Array3D<double> Dx;
  Array3D<double> Dy;
  Array3D<double> Dz;

  Array3D<double> Hx;
  Array3D<double> Hy;
  Array3D<double> Hz;

  Array3D<double> Bx;
  Array3D<double> By;
  Array3D<double> Bz;

#ifdef _MPI
  MPI_Comm mpi_comm;
#endif

	double E() const;
};
