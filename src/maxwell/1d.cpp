#include "1d.h"
#include "../const.h"

maxwell::Workspace1D::Workspace1D(Grid1d const& grid):
	grid(grid),
	E(grid),
	D(grid),
	H(grid) {
}

maxwell::Workspace1D::~Workspace1D() {}

void maxwell::Workspace1D::prop_Bz(Arr1& Bz, Arr1 const& Ey, double ksi) const {
#pragma omp for
	for (int i=0; i<Bz.grid.n-1; i++) {
		Bz(i) = Bz(i) - ksi*(Ey(i+1) - Ey(i));
	}

	{
		int i = grid.n-1;
		Bz(i) = -ksi*(0.0 - Ey(i));
	}
}

void maxwell::Workspace1D::prop_Dy(Arr1& Dy, Arr1 const& Hz, double ksi) const {
	{
		int i = 0;
		Dy(i) = -ksi*(Hz(i) - 0.0);
	}

#pragma omp for
	for (int i=1; i<Dy.grid.n; i++) {
		Dy(i) = Dy(i) - ksi*(Hz(i) - Hz(i-1));
	}
}

void maxwell::Workspace1D::prop(double dt) {
	double ksi = C_au * dt / grid.d;

	prop_Bz(H, E, ksi);
	prop_Dy(E, H, ksi);
}

void maxwell::Workspace1D::prop(double dt, Arr1 const& eps) {
	double ksi = C_au * dt / grid.d;

	prop_Bz(H, E, ksi);
	prop_Dy(D, H, ksi);

#pragma omp parallel for
	for (int i=0; i<grid.n; i++) {
		E(i) = D(i) / eps(i);
	}
}

void maxwell::Workspace1D::prop_pol(double dt, Arr1 const& P) {
	double ksi = C_au * dt / grid.d;

	prop_Bz(H, E, ksi);
	prop_Dy(D, H, ksi);

#pragma omp parallel for
	for (int i=0; i<grid.n; i++) {
		E(i) = D(i) - 4*M_PI*P(i);
	}
}

int maxwell::Workspace1D::move_center_window_to_max_E() {
	int center = E.argmax([](double x) -> double {
			return abs(x);
			});

	int shift_value = center - grid.n/2;
	E.shift(shift_value, 0.0);
	H.shift(shift_value, 0.0);
	D.shift(shift_value, 0.0);

	return shift_value;
}

maxwell::Workspace2D::Workspace2D(Grid2d const& grid):
	grid(grid),
	Ez(grid),
	Dz(grid),
	Hx(grid),
	Hy(grid) {
}

maxwell::Workspace2D::~Workspace2D() {}

void maxwell::Workspace2D::prop_Hx(Arr2& Hx, Arr2 const& Ez, double ksi) {
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

void maxwell::Workspace2D::prop_Hy(Arr2& Hy, Arr2 const& Ez, double ksi) {
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

void maxwell::Workspace2D::prop_Dz(Arr2& Dz, Arr2 const& Hx, Arr2 const& Hy, double ksi) {
	Grid2d const& grid = Dz.grid;

#pragma omp parallel for collapse(2)
	for (int ix=1; ix<grid.n[iX]; ix++) {
		for (int iy=1; iy<grid.n[iY]; iy++) {
			Dz(ix, iy) += ksi*(Hy(ix, iy) - Hy(ix-1, iy) -
					           Hx(ix, iy) - Hx(ix, iy-1));
		}
	}
}

void maxwell::Workspace2D::prop_Ez(Arr2& Ez, Arr2 const& Dz, Arr2 const& eps) {
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

void maxwell::Workspace2D::prop(double dt, Arr2 const& eps) {
}
