#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "sphere_kn.h"
#include "jrcd.h"

#include "utils.h"

double const M_PI = 3.14;

double U(double r) {
	return -1.0/r;
}

// dUdz / cos(\theta)
double dUdz(double r) {
	return 1.0/pow(r, 2);
}

double const omega = 4*5.6e-2;
double const E0 = 5.34e-2;

int main(int argc, char *argv[]) {
	sphere_grid_t grid;
	grid.Nr = 1000;
	grid.Nl = 20;
	grid.dr = 0.2;

	double const T = 2.0*M_PI/omega;

	double E(double t) {
		return E0*sin(omega*t)*smoothpulse(t, T, 2*T);
	}

	double dt = 0.1;

	sphere_kn_workspace_t* ws = sphere_kn_workspace_alloc(&grid, dt, U, E);
	sphere_wavefunc_t* psi = sphere_wavefunc_alloc(&grid, 0);

	for (int i = 0; i < grid.Nr; ++i) {
		double r = (i+1)*grid.dr;
		psi->data[i]         = 2*r*exp(-r);
	}

	int Nt = (int)(12*T/dt);
	double a[Nt];
	double t = 0.0;
	for (int i = 0; i < Nt; ++i) {
		a[i] = az(psi, E, dUdz, t);
		printf("%f\n", a[i]);

		sphere_kn_workspace_prop(ws, psi, t);
		sphere_wavefunc_normalize(psi);

		t += dt;
	}

	sphere_kn_workspace_free(ws);
	sphere_wavefunc_free(psi);
	
	return 0;
}
