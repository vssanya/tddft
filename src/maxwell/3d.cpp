#include "3d.h"
#include "../const.h"
#include "mpi.h"


maxwell::Workspace3D::Workspace3D(Grid3d const& grid, MPI_Comm mpi_comm):
	grid(grid),
	bound_grid(grid.getSliceZ()),
	rb_Ex(bound_grid),
	rb_Ey(bound_grid),
	lb_Hx(bound_grid),
	lb_Hy(bound_grid),
	mpi_comm(mpi_comm)
{
#ifdef _MPI
		if (mpi_comm != MPI_COMM_NULL) {
			MPI_Comm_size(mpi_comm, &mpi_size);
			MPI_Comm_rank(mpi_comm, &mpi_rank);
		}
#endif
}

maxwell::Workspace3D::~Workspace3D() {}

void maxwell::Workspace3D::prop_E(Field3D& f, double dt) {
	double ksi_x = C_au * dt / grid.d[0];
	double ksi_y = C_au * dt / grid.d[1];
	double ksi_z = C_au * dt / grid.d[2];

#ifdef _MPI
	MPI_Request request[4] = {MPI_REQUEST_NULL};
	const int X_TAG = 0;
	const int Y_TAG = 1;

	if (mpi_comm != MPI_COMM_NULL && mpi_size > 1) {
		if (mpi_rank != mpi_size - 1) {
			MPI_Isend(&f.Hx(0,0,grid.n[2]-1), bound_grid.size(), MPI_DOUBLE, mpi_rank+1, X_TAG, mpi_comm, &request[0]);
			MPI_Isend(&f.Hy(0,0,grid.n[2]-1), bound_grid.size(), MPI_DOUBLE, mpi_rank+1, Y_TAG, mpi_comm, &request[1]);
		}

		if (mpi_rank != 0) {
			MPI_Irecv(&lb_Hx(0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank-1, X_TAG, mpi_comm, &request[2]);
			MPI_Irecv(&lb_Hy(0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank-1, Y_TAG, mpi_comm, &request[3]);
		}
	}
#endif

#pragma omp parallel for collapse(3)
	for (int iz=1; iz<grid.n[2]; iz++) {
		for (int iy=1; iy<grid.n[1]; iy++) {
			for (int ix=1; ix<grid.n[0]; ix++) {
				f.Ex(ix, iy, iz) += + ksi_y*(f.Hz(ix, iy, iz) - f.Hz(ix, iy-1, iz))
					                  - ksi_z*(f.Hy(ix, iy, iz) - f.Hy(ix, iy, iz-1));

				f.Ey(ix, iy, iz) += + ksi_z*(f.Hx(ix, iy, iz) - f.Hx(ix, iy, iz-1))
					                  - ksi_x*(f.Hz(ix, iy, iz) - f.Hz(ix-1, iy, iz));

				f.Ez(ix, iy, iz) += + ksi_x*(f.Hy(ix, iy, iz) - f.Hy(ix-1, iy, iz))
					                  - ksi_y*(f.Hx(ix, iy, iz) - f.Hx(ix, iy-1, iz));
			}
		}
	}

#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL && mpi_size > 1) {
		MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

		{
			int iz=0;
#pragma omp parallel for collapse(2)
			for (int iy=1; iy<grid.n[1]; iy++) {
				for (int ix=1; ix<grid.n[0]; ix++) {
					f.Ex(ix, iy, iz) += + ksi_y*(f.Hz(ix, iy, iz) - f.Hz(ix, iy-1, iz))
						- ksi_z*(f.Hy(ix, iy, iz) - lb_Hy(ix, iy));

					f.Ey(ix, iy, iz) += + ksi_z*(f.Hx(ix, iy, iz) - lb_Hx(ix, iy))
						- ksi_x*(f.Hz(ix, iy, iz) - f.Hz(ix-1, iy, iz));

					f.Ez(ix, iy, iz) += + ksi_x*(f.Hy(ix, iy, iz) - f.Hy(ix-1, iy, iz))
						- ksi_y*(f.Hx(ix, iy, iz) - f.Hx(ix, iy-1, iz));
				}
			}
		}
	}
#endif
}

void maxwell::Workspace3D::prop_E(Field3D& f, double dt, Array3D<double>& sigma) {
	double ksi_x = C_au * dt / grid.d[0];
	double ksi_y = C_au * dt / grid.d[1];
	double ksi_z = C_au * dt / grid.d[2];

#ifdef _MPI
	MPI_Request request[4] = {MPI_REQUEST_NULL};
	const int X_TAG = 0;
	const int Y_TAG = 1;

	if (mpi_comm != MPI_COMM_NULL && mpi_size > 1) {
		if (mpi_rank != mpi_size - 1) {
			MPI_Isend(&f.Hx(0,0,grid.n[2]-1), bound_grid.size(), MPI_DOUBLE, mpi_rank+1, X_TAG, mpi_comm, &request[0]);
			MPI_Isend(&f.Hy(0,0,grid.n[2]-1), bound_grid.size(), MPI_DOUBLE, mpi_rank+1, Y_TAG, mpi_comm, &request[1]);
		}

		if (mpi_rank != 0) {
			MPI_Irecv(&lb_Hx(0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank-1, X_TAG, mpi_comm, &request[2]);
			MPI_Irecv(&lb_Hy(0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank-1, Y_TAG, mpi_comm, &request[3]);
		}
	}
#endif

#pragma omp parallel for collapse(3)
	for (int iz=1; iz<grid.n[2]; iz++) {
		for (int iy=1; iy<grid.n[1]; iy++) {
			for (int ix=1; ix<grid.n[0]; ix++) {
				double const Dl[3] = {f.Dx(ix, iy, iz), f.Dy(ix, iy, iz), f.Dz(ix, iy, iz)};

				f.Dx(ix, iy, iz) += + ksi_y*(f.Hz(ix, iy, iz) - f.Hz(ix, iy-1, iz))
					                - ksi_z*(f.Hy(ix, iy, iz) - f.Hy(ix, iy, iz-1));

				f.Dy(ix, iy, iz) += + ksi_z*(f.Hx(ix, iy, iz) - f.Hx(ix, iy, iz-1))
					                - ksi_x*(f.Hz(ix, iy, iz) - f.Hz(ix-1, iy, iz));

				f.Dz(ix, iy, iz) += + ksi_x*(f.Hy(ix, iy, iz) - f.Hy(ix-1, iy, iz))
					                - ksi_y*(f.Hx(ix, iy, iz) - f.Hx(ix, iy-1, iz));

				double nu = 2*M_PI*dt*sigma(ix, iy, iz);

				f.Ex(ix, iy, iz) = f.Ex(ix, iy, iz)*(1 - nu)/(1 + nu) + (f.Dx(ix, iy, iz) - Dl[0])/(1 + nu);
				f.Ey(ix, iy, iz) = f.Ey(ix, iy, iz)*(1 - nu)/(1 + nu) + (f.Dy(ix, iy, iz) - Dl[1])/(1 + nu);
				f.Ez(ix, iy, iz) = f.Ez(ix, iy, iz)*(1 - nu)/(1 + nu) + (f.Dz(ix, iy, iz) - Dl[2])/(1 + nu);
			}
		}
	}

#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL && mpi_size > 1) {
		MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

		{
			int iz=0;
#pragma omp parallel for collapse(2)
			for (int iy=1; iy<grid.n[1]; iy++) {
				for (int ix=1; ix<grid.n[0]; ix++) {
					double const Dl[3] = {f.Dx(ix, iy, iz), f.Dy(ix, iy, iz), f.Dz(ix, iy, iz)};

					f.Dx(ix, iy, iz) += + ksi_y*(f.Hz(ix, iy, iz) - f.Hz(ix, iy-1, iz))
						- ksi_z*(f.Hy(ix, iy, iz) - lb_Hy(ix, iy));

					f.Dy(ix, iy, iz) += + ksi_z*(f.Hx(ix, iy, iz) - lb_Hx(ix, iy))
						- ksi_x*(f.Hz(ix, iy, iz) - f.Hz(ix-1, iy, iz));

					f.Dz(ix, iy, iz) += + ksi_x*(f.Hy(ix, iy, iz) - f.Hy(ix-1, iy, iz))
						- ksi_y*(f.Hx(ix, iy, iz) - f.Hx(ix, iy-1, iz));

					double nu = 2*M_PI*dt*sigma(ix, iy, iz);

					f.Ex(ix, iy, iz) = f.Ex(ix, iy, iz)*(1 - nu)/(1 + nu) + (f.Dx(ix, iy, iz) - Dl[0])/(1 + nu);
					f.Ey(ix, iy, iz) = f.Ey(ix, iy, iz)*(1 - nu)/(1 + nu) + (f.Dy(ix, iy, iz) - Dl[1])/(1 + nu);
					f.Ez(ix, iy, iz) = f.Ez(ix, iy, iz)*(1 - nu)/(1 + nu) + (f.Dz(ix, iy, iz) - Dl[2])/(1 + nu);
				}
			}
		}
	}
#endif
}

void maxwell::Workspace3D::prop_E(Field3D& f, double dt, Array3D<double> j[3]) {
	double ksi_x = C_au * dt / grid.d[0];
	double ksi_y = C_au * dt / grid.d[1];
	double ksi_z = C_au * dt / grid.d[2];

#ifdef _MPI
	MPI_Request request[4] = {MPI_REQUEST_NULL};
	const int X_TAG = 0;
	const int Y_TAG = 1;

	if (mpi_comm != MPI_COMM_NULL && mpi_size > 1) {
		if (mpi_rank != mpi_size - 1) {
			MPI_Isend(&f.Hx(0,0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank+1, X_TAG, mpi_comm, &request[0]);
			MPI_Isend(&f.Hy(0,0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank+1, Y_TAG, mpi_comm, &request[1]);
		}

		if (mpi_rank != 0) {
			MPI_Irecv(&lb_Hx(0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank-1, X_TAG, mpi_comm, &request[2]);
			MPI_Irecv(&lb_Hy(0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank-1, Y_TAG, mpi_comm, &request[3]);
		}
	}
#endif

#pragma omp parallel for collapse(3)
	for (int iz=1; iz<grid.n[2]; iz++) {
		for (int iy=1; iy<grid.n[1]; iy++) {
			for (int ix=1; ix<grid.n[0]; ix++) {
				f.Ex(ix, iy, iz) += + ksi_y*(f.Hz(ix, iy, iz) - f.Hz(ix, iy-1, iz))
					                - ksi_z*(f.Hy(ix, iy, iz) - f.Hy(ix, iy, iz-1))
													- 4*M_PI/C_au*dt*j[0](ix, iy, iz);

				f.Ey(ix, iy, iz) += + ksi_z*(f.Hx(ix, iy, iz) - f.Hx(ix, iy, iz-1))
					                - ksi_x*(f.Hz(ix, iy, iz) - f.Hz(ix-1, iy, iz))
													- 4*M_PI/C_au*dt*j[0](ix, iy, iz);

				f.Ez(ix, iy, iz) += + ksi_x*(f.Hy(ix, iy, iz) - f.Hy(ix-1, iy, iz))
					                - ksi_y*(f.Hx(ix, iy, iz) - f.Hx(ix, iy-1, iz))
													- 4*M_PI/C_au*dt*j[0](ix, iy, iz);
			}
		}
	}


#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL && mpi_size > 1) {
		MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

		{
			int iz=0;
#pragma omp parallel for collapse(2)
			for (int iy=1; iy<grid.n[1]; iy++) {
				for (int ix=1; ix<grid.n[0]; ix++) {
					f.Ex(ix, iy, iz) += + ksi_y*(f.Hz(ix, iy, iz) - f.Hz(ix, iy-1, iz))
						- ksi_z*(f.Hy(ix, iy, iz) - lb_Hy(ix, iy))
						- 4*M_PI/C_au*dt*j[0](ix, iy, iz);

					f.Ey(ix, iy, iz) += + ksi_z*(f.Hx(ix, iy, iz) - lb_Hx(ix, iy))
						- ksi_x*(f.Hz(ix, iy, iz) - f.Hz(ix-1, iy, iz))
						- 4*M_PI/C_au*dt*j[0](ix, iy, iz);

					f.Ez(ix, iy, iz) += + ksi_x*(f.Hy(ix, iy, iz) - f.Hy(ix-1, iy, iz))
						- ksi_y*(f.Hx(ix, iy, iz) - f.Hx(ix, iy-1, iz))
						- 4*M_PI/C_au*dt*j[0](ix, iy, iz);
				}
			}
		}
	}
#endif
}

void maxwell::Workspace3D::prop_H(Field3D& f, double dt) {
	double ksi_x = C_au * dt / grid.d[0];
	double ksi_y = C_au * dt / grid.d[1];
	double ksi_z = C_au * dt / grid.d[2];

#ifdef _MPI
	MPI_Request request[4] = {MPI_REQUEST_NULL};
	const int X_TAG = 0;
	const int Y_TAG = 1;

	if (mpi_comm != MPI_COMM_NULL && mpi_size > 1) {
		if (mpi_rank != 0) {
			MPI_Isend(&f.Ex(0,0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank-1, X_TAG, mpi_comm, &request[0]);
			MPI_Isend(&f.Ey(0,0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank-1, Y_TAG, mpi_comm, &request[1]);
		}

		if (mpi_rank != mpi_size - 1) {
			MPI_Irecv(&rb_Ex(0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank+1, X_TAG, mpi_comm, &request[2]);
			MPI_Irecv(&rb_Ey(0,0), bound_grid.size(), MPI_DOUBLE, mpi_rank+1, Y_TAG, mpi_comm, &request[3]);
		}
	}
#endif

#pragma omp parallel for collapse(3)
	for (int iz=0; iz<grid.n[2]-1; iz++) {
		for (int iy=0; iy<grid.n[1]-1; iy++) {
			for (int ix=0; ix<grid.n[0]-1; ix++) {
				f.Hx(ix, iy, iz) += - ksi_y*(f.Ez(ix, iy+1, iz) - f.Ez(ix, iy, iz))
					                + ksi_z*(f.Ey(ix, iy, iz+1) - f.Ey(ix, iy, iz));

				f.Hy(ix, iy, iz) += - ksi_z*(f.Ex(ix, iy, iz+1) - f.Ex(ix, iy, iz))
					                + ksi_x*(f.Ez(ix+1, iy, iz) - f.Ez(ix, iy, iz));

				f.Hz(ix, iy, iz) += - ksi_x*(f.Ey(ix+1, iy, iz) - f.Ey(ix, iy, iz))
					                + ksi_y*(f.Ex(ix, iy+1, iz) - f.Ex(ix, iy, iz));
			}
		}
	}

#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL && mpi_size > 1) {
		MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

		{
			int iz=grid.n[2]-1;
#pragma omp parallel for collapse(2)
			for (int iy=0; iy<grid.n[1]-1; iy++) {
				for (int ix=0; ix<grid.n[0]-1; ix++) {
					f.Hx(ix, iy, iz) += - ksi_y*(f.Ez(ix, iy+1, iz) - f.Ez(ix, iy, iz))
						+ ksi_z*(rb_Ey(ix, iy) - f.Ey(ix, iy, iz));

					f.Hy(ix, iy, iz) += - ksi_z*(rb_Ex(ix, iy) - f.Ex(ix, iy, iz))
						+ ksi_x*(f.Ez(ix+1, iy, iz) - f.Ez(ix, iy, iz));

					f.Hz(ix, iy, iz) += - ksi_x*(f.Ey(ix+1, iy, iz) - f.Ey(ix, iy, iz))
						+ ksi_y*(f.Ex(ix, iy+1, iz) - f.Ex(ix, iy, iz));
				}
			}
		}
	}
#endif
}
