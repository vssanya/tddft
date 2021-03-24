#include "3d.h"
#include "../const.h"


maxwell::Workspace3D::Workspace3D(Grid3d const& grid):
	grid(grid),
	Ex(grid),
	Ey(grid),
	Ez(grid),
	Hx(grid),
	Hy(grid),
	Hz(grid) {
}

maxwell::Workspace3D::~Workspace3D() {}

void maxwell::Workspace3D::prop_E(double dt) {
	double ksi_x = C_au * dt / grid.d[0];
	double ksi_y = C_au * dt / grid.d[1];
	double ksi_z = C_au * dt / grid.d[2];

#pragma omp parallel for collapse(3)
	for (int ix=1; ix<grid.n[0]; ix++) {
		for (int iy=1; iy<grid.n[0]; iy++) {
			for (int iz=1; iz<grid.n[0]; iz++) {
				Ex(ix, iy, iz) += + ksi_z*(Hz(ix, iy, iz) - Hz(ix, iy-1, iz))
					                - ksi_y*(Hy(ix, iy, iz) - Hy(ix, iy, iz-1));

				Ey(ix, iy, iz) += + ksi_x*(Hx(ix, iy, iz) - Hx(ix, iy, iz-1))
					                - ksi_z*(Hz(ix, iy, iz) - Hz(ix-1, iy, iz));

				Ez(ix, iy, iz) += + ksi_y*(Hy(ix, iy, iz) - Hy(ix-1, iy, iz))
					                - ksi_x*(Hx(ix, iy, iz) - Hx(ix, iy, iz-1));
			}
		}
	}
}

void maxwell::Workspace3D::prop_E(double dt, Arr3 j[3]) {
	double ksi_x = C_au * dt / grid.d[0];
	double ksi_y = C_au * dt / grid.d[1];
	double ksi_z = C_au * dt / grid.d[2];

#pragma omp parallel for collapse(3)
	for (int ix=1; ix<grid.n[0]; ix++) {
		for (int iy=1; iy<grid.n[0]; iy++) {
			for (int iz=1; iz<grid.n[0]; iz++) {
				Ex(ix, iy, iz) += + ksi_z*(Hz(ix, iy, iz) - Hz(ix, iy-1, iz))
					                - ksi_y*(Hy(ix, iy, iz) - Hy(ix, iy, iz-1))
													- 4*M_PI/C_au*dt*j[0](ix, iy, iz);

				Ey(ix, iy, iz) += + ksi_x*(Hx(ix, iy, iz) - Hx(ix, iy, iz-1))
					                - ksi_z*(Hz(ix, iy, iz) - Hz(ix-1, iy, iz))
													- 4*M_PI/C_au*dt*j[0](ix, iy, iz);

				Ez(ix, iy, iz) += + ksi_y*(Hy(ix, iy, iz) - Hy(ix-1, iy, iz))
					                - ksi_x*(Hx(ix, iy, iz) - Hx(ix, iy, iz-1))
													- 4*M_PI/C_au*dt*j[0](ix, iy, iz);
			}
		}
	}
}

void maxwell::Workspace3D::prop_H(double dt) {
	double ksi_x = C_au * dt / grid.d[0];
	double ksi_y = C_au * dt / grid.d[1];
	double ksi_z = C_au * dt / grid.d[2];

#pragma omp parallel for collapse(3)
	for (int ix=0; ix<grid.n[0]-1; ix++) {
		for (int iy=0; iy<grid.n[0]-1; iy++) {
			for (int iz=0; iz<grid.n[0]-1; iz++) {
				Hx(ix, iy, iz) += - ksi_z*(Ez(ix, iy+1, iz) - Ez(ix, iy, iz))
					                + ksi_y*(Ey(ix, iy, iz+1) - Ey(ix, iy, iz));

				Hy(ix, iy, iz) += - ksi_x*(Ex(ix, iy, iz+1) - Ex(ix, iy, iz))
					                + ksi_z*(Ez(ix+1, iy, iz) - Ez(ix, iy, iz));

				Hz(ix, iy, iz) += - ksi_y*(Ey(ix+1, iy, iz) - Ey(ix, iy, iz)) -
					                + ksi_x*(Ex(ix, iy, iz+1) - Ex(ix, iy, iz));
			}
		}
	}
}
