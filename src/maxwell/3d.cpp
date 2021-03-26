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
		for (int iy=1; iy<grid.n[1]; iy++) {
			for (int iz=1; iz<grid.n[2]; iz++) {
				Ex(ix, iy, iz) += + ksi_y*(Hz(ix, iy, iz) - Hz(ix, iy-1, iz))
					                - ksi_z*(Hy(ix, iy, iz) - Hy(ix, iy, iz-1));

				Ey(ix, iy, iz) += + ksi_z*(Hx(ix, iy, iz) - Hx(ix, iy, iz-1))
					                - ksi_x*(Hz(ix, iy, iz) - Hz(ix-1, iy, iz));

				Ez(ix, iy, iz) += + ksi_x*(Hy(ix, iy, iz) - Hy(ix-1, iy, iz))
					                - ksi_y*(Hx(ix, iy, iz) - Hx(ix, iy-1, iz));
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
		for (int iy=1; iy<grid.n[1]; iy++) {
			for (int iz=1; iz<grid.n[2]; iz++) {
				Ex(ix, iy, iz) += + ksi_y*(Hz(ix, iy, iz) - Hz(ix, iy-1, iz))
					                - ksi_z*(Hy(ix, iy, iz) - Hy(ix, iy, iz-1))
													- 4*M_PI/C_au*dt*j[0](ix, iy, iz);

				Ey(ix, iy, iz) += + ksi_z*(Hx(ix, iy, iz) - Hx(ix, iy, iz-1))
					                - ksi_x*(Hz(ix, iy, iz) - Hz(ix-1, iy, iz))
													- 4*M_PI/C_au*dt*j[0](ix, iy, iz);

				Ez(ix, iy, iz) += + ksi_x*(Hy(ix, iy, iz) - Hy(ix-1, iy, iz))
					                - ksi_y*(Hx(ix, iy, iz) - Hx(ix, iy-1, iz))
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
		for (int iy=0; iy<grid.n[1]-1; iy++) {
			for (int iz=0; iz<grid.n[2]-1; iz++) {
				Hx(ix, iy, iz) += - ksi_y*(Ez(ix, iy+1, iz) - Ez(ix, iy, iz))
					                + ksi_z*(Ey(ix, iy, iz+1) - Ey(ix, iy, iz));

				Hy(ix, iy, iz) += - ksi_z*(Ex(ix, iy, iz+1) - Ex(ix, iy, iz))
					                + ksi_x*(Ez(ix+1, iy, iz) - Ez(ix, iy, iz));

				Hz(ix, iy, iz) += - ksi_x*(Ey(ix+1, iy, iz) - Ey(ix, iy, iz))
					                + ksi_y*(Ex(ix, iy+1, iz) - Ex(ix, iy, iz));
			}
		}
	}
}
