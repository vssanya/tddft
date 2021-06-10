#include "field_3d.h"

Grid3d getLocalGrid(Grid3d const& grid, MPI_Comm mpi_comm) {
#ifdef _MPI
	if (mpi_comm == MPI_COMM_NULL) {
		return grid;
	} else {
		int size;
		int rank;

		MPI_Comm_size(mpi_comm, &size);
		MPI_Comm_rank(mpi_comm, &rank);

		auto grid_new = Grid3d(grid);
		grid_new.n[2] = grid.n[2] / size + (grid.n[2] % size > rank + 1 ? 1 : 0);
		return grid_new;
	}
#else
	return grid;
#endif
}

Field3D::Field3D(Grid3d const& grid, MPI_Comm mpi_comm):
	grid(getLocalGrid(grid, mpi_comm)),
	Ex(this->grid),
	Ey(this->grid),
	Ez(this->grid),
	Hx(this->grid),
	Hy(this->grid),
	Hz(this->grid),
	mpi_comm(mpi_comm)
{
}
