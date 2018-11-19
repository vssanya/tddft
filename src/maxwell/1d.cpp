#include "1d.h"
#include "../const.h"

maxwell::Workspace1D::Workspace1D(Grid1d const& grid):
	grid(grid)
{
	E = new double[grid.n];
	D = new double[grid.n];
	H = new double[grid.n];
}

maxwell::Workspace1D::~Workspace1D() {
	delete[] E;
	delete[] D;
	delete[] H;
}

void prop_Bz(double* Bz, double* Ey, Grid1d const& grid, double ksi) {
#pragma omp for
	for (int i=0; i<grid.n-1; i++) {
		Bz[i] = Bz[i] - ksi*(Ey[i+1] - Ey[i]);
	}

	{
		int i = grid.n-1;
		Bz[i] = -ksi*(0.0 - Ey[i]);
	}
}

void prop_Dy(double* Dy, double* Hz, Grid1d const& grid, double ksi) {
	{
		int i = 0;
		Dy[i] = -ksi*(Hz[i] - 0.0);
	}

#pragma omp for
	for (int i=1; i<grid.n; i++) {
		Dy[i] = Dy[i] - ksi*(Hz[i] - Hz[i-1]);
	}
}

void maxwell::Workspace1D::prop(double dt) {
	double ksi = C_au * dt / grid.d;

	prop_Bz(H, E, grid, ksi);
	prop_Dy(E, H, grid, ksi);
}

void maxwell::Workspace1D::prop(double dt, double eps[]) {
	double ksi = C_au * dt / grid.d;

	prop_Bz(H, E, grid, ksi);
	prop_Dy(D, H, grid, ksi);

	for (int i=0; i<grid.n; i++) {
		E[i] = D[i] / eps[i];
	}
}

maxwell::Workspace2D::Workspace2D(Grid2d const& grid): grid(grid), Ez(grid), Dz(grid), Hx(grid), Hy(grid) {
}

maxwell::Workspace2D::~Workspace2D() {}

void maxwell::Workspace2D::prop_Hx(Array2D& Hx, Array2D const& Ez, double ksi) {
	Grid2d const& grid = Hx.grid;

#pragma omp parallel for collapse(2)
	for (int ix=0; ix<grid.n[iX]; ix++) {
		for (int iy=0; iy<grid.n[iY] - 1; iy++) {
			Hx(ix, iy) -= ksi*(Ez(ix, iy+1) - Ez(ix, iy));
		}
	}

#pragma omp parallel for
	for (int ix=0; ix<grid.n[iX]; ix++) {
		int iy = grid.n[iY] - 1;
		Hx(ix, iy) -= ksi*(0.0 - Ez(ix, iy));
	}
}

void maxwell::Workspace2D::prop_Hy(Array2D& Hy, Array2D const& Ez, double ksi) {
	Grid2d const& grid = Hy.grid;

#pragma omp parallel for collapse(2)
	for (int ix=0; ix<grid.n[iX]-1; ix++) {
		for (int iy=0; iy<grid.n[iY]; iy++) {
			Hy(ix, iy) += ksi*(Ez(ix+1, iy) - Ez(ix, iy));
		}
	}

	{
		int ix = grid.n[iX]-1;
#pragma omp parallel for
		for (int iy=0; iy<grid.n[iY]; iy++) {
			Hy(ix, iy) += ksi*(0.0 - Ez(ix, iy));
		}
	}
}

void maxwell::Workspace2D::prop_Dz(Array2D& Dz, Array2D const& Hx, Array2D const& Hy, double ksi) {
	Grid2d const& grid = Dz.grid;

#pragma omp parallel for collapse(2)
	for (int ix=1; ix<grid.n[iX]; ix++) {
		for (int iy=1; iy<grid.n[iY]; iy++) {
			Dz(ix, iy) += ksi*(Hy(ix, iy) - Hy(ix-1, iy) -
					           Hx(ix, iy) - Hx(ix, iy-1));
		}
	}
}

void maxwell::Workspace2D::prop_Ez(Array2D& Ez, Array2D const& Dz, Array2D const& eps) {
	Grid2d const& grid = Ez.grid;

#pragma omp parallel for collapse(2)
	for (int ix=0; ix<grid.n[iX]; ix++) {
		for (int iy=0; iy<grid.n[iY]; iy++) {
			Ez(ix, iy) = Dz(ix, iy) / eps(ix, iy);
		}
	}
}

void maxwell::Workspace2D::prop(double dt) {
	double ksi = C_au * dt / grid.d[iX];

	prop_Hx(Hx, Ez, ksi);
	prop_Hy(Hy, Ez, ksi);
	prop_Dz(Ez, Hx, Hy, ksi);
}

void maxwell::Workspace2D::prop(double dt, double eps[]) {

}
