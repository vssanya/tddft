#include "1d.h"
#include "../const.h"

maxwell::WorkspaceCyl1D::WorkspaceCyl1D(Grid1d const& grid):
	grid(grid),
	Er(grid),
	Ephi(grid),
	Hz(grid),
	jr(grid),
	jphi(grid)
{
}

maxwell::WorkspaceCyl1D::~WorkspaceCyl1D() {}

void maxwell::WorkspaceCyl1D::prop(double dt) {
	double ksi = C_au * dt / grid.d;

	// Ephi = Ephi / r
	{
		int i = 0;
		Ephi(i) += -4*M_PI*jphi(i)*dt - ksi*2*Hz(i);
	}

#pragma omp parallel for
	for (int i=1; i<grid.n; i++) {
		Ephi(i) += -4*M_PI*jphi(i)*dt - ksi*(Hz(i) - Hz(i-1));
	}

#pragma omp parallel for
	for (int i=0; i<grid.n; i++) {
		double r = grid.x(i) + grid.d/2;
		Er(i) += - 4*M_PI*jr(i)*dt + C_au*dt*Hz(i) / r;
	}

#pragma omp parallel for
	for (int i=0; i<grid.n-1; i++) {
		double r = grid.x(i) + grid.d/2;
		double rp = grid.x(i);
		double dr = grid.d;
		Hz(i) += - dt*C_au*(((rp + dr)*Ephi(i+1) - rp*Ephi(i))/dr + Er(i)) / r;
	}
}

void maxwell::WorkspaceCyl1D::prop(double dt, double* N, double nu) {
	double ksi = C_au * dt / grid.d;

	// Ephi = Ephi / r
	{
		int i = 0;

		double Ephi_l = Ephi(i);

		double ap = 1 + 0.5*dt*nu;
		double am = 2 - ap;

		double bp = 1 + M_PI*dt*dt*N[2*i]/ap;
		double bm = 2 - bp;

		Ephi(i) = (Ephi(i)*bm - 4*M_PI*jphi(i)*dt/ap - 2*ksi*Hz(i))/bp;
		jphi(i) = jphi(i)*am + 0.5*dt*N[2*i]*(Ephi(i) + Ephi_l);
	}

#pragma omp parallel for
	for (int i=1; i<grid.n; i++) {
		double Ephi_l = Ephi(i);

		double ap = 1 + 0.5*dt*nu;
		double am = 2 - ap;

		double bp = 1 + M_PI*dt*dt*N[2*i]/ap;
		double bm = 2 - bp;

		Ephi(i) = (Ephi(i)*bm - 4*M_PI*jphi(i)*dt/ap - ksi*(Hz(i) - Hz(i-1)))/bp;
		jphi(i) = (jphi(i)*am + 0.5*dt*N[2*i]*(Ephi(i) + Ephi_l))/ap;
	}

#pragma omp parallel for
	for (int i=0; i<grid.n; i++) {
		double r = grid.x(i) + grid.d/2;

		double Er_l = Er(i);

		double ap = 1 + 0.5*dt*nu;
		double am = 2 - ap;

		double bp = 1 + M_PI*dt*dt*N[2*i+1]/ap;
		double bm = 2 - bp;

		Er(i) = (Er(i)*bm - 4*M_PI*jr(i)*dt/ap + C_au*dt*Hz(i) / r)/bp;
		jr(i) = (jr(i)*am + 0.5*dt*N[2*i+1]*(Er(i) + Er_l))/ap;
	}

#pragma omp parallel for
	for (int i=0; i<grid.n-1; i++) {
		double r = grid.x(i) + grid.d/2;
		double rp = grid.x(i);
		double dr = grid.d;
		Hz(i) += - dt*C_au*(((rp + dr)*Ephi(i+1) - rp*Ephi(i))/dr + Er(i)) / r;
	}
}
