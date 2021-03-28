#include "1d.h"
#include "../const.h"

maxwell::WorkspaceCyl1D::WorkspaceCyl1D(Grid1d const& grid):
	grid(grid),
	Er(grid),
	Ephi(grid),
	Hz(grid) {
}

maxwell::WorkspaceCyl1D::~WorkspaceCyl1D() {}

void maxwell::WorkspaceCyl1D::prop(double dt, double* j) {
	double ksi = C_au * dt / grid.d;

	// Ephi = Ephi / r
	{
		int i = 0;
		Ephi(i) += 4*M_PI*j[2*i]*dt - ksi*2*Hz(i);
	}

#pragma omp parallel for
	for (int i=1; i<grid.n; i++) {
		Ephi(i) += 4*M_PI*j[2*i]*dt - ksi*(Hz(i) - Hz(i-1));
	}

#pragma omp parallel for
	for (int i=0; i<grid.n; i++) {
		double r = grid.x(i) + grid.d/2;
		Er(i) += - 4*M_PI*j[2*i+1]*dt + C_au*dt*Hz(i) / r;
	}

#pragma omp parallel for
	for (int i=0; i<grid.n-1; i++) {
		double r = grid.x(i) + grid.d/2;
		double rp = grid.x(i);
		double dr = grid.d;
		Hz(i) += - dt*C_au*(((rp + dr)*Ephi(i+1) - rp*Ephi(i))/dr + Er(i)) / r;
		//Hz(i) += - ksi*(Ephi(i+1) - Ephi(i)) - dt*C_au*(0.5*(Ephi(i+1) + Ephi(i)) + Er(i)) / r;
		//Hz(i) += - dt*C_au*((Ephi(i+1) - Ephi(i))/dr + Er(i)) / r;
	}
}
