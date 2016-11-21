#include <stdlib.h>
#include <stdio.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "sphere_kn.h"
#include "jrcd.h"

#include "utils.h"

#include "fourier.h"

double U(double r) {
	return -1.0/r;
}

// dUdz / cos(\theta)
double dUdz(double r) {
	return 1.0/pow(r, 2);
}

int main() {
	sphere_grid_t grid = {
		.Nr = 2000,
		.Nl = 80,
		.dr = 0.125
	};

	double const omega = 5.6e-2;
	double const E0 = 3.77e-2;
	double const T = 2.0*M_PI/omega;
	double const tp = T*2.05/2.67/(2*log(2));

	double Uabs(double r) {
		double r_max = grid.dr*grid.Nr;
		double dr = 0.2*r_max;
		return 2*smoothstep(r, r_max-dr, r_max);
	}

	double dt = 0.025;
	int Nt = (int)(3*T/dt);

	double dw = 20*omega/Nt;
	cdouble P[Nt];
	for (int i = 0; i < Nt; ++i) {
		P[i] = 0.0;
	}

	sphere_kn_workspace_t* ws = sphere_kn_workspace_alloc(&grid, dt, U, Uabs);
	sphere_wavefunc_t* psi = sphere_wavefunc_alloc(&grid, 0);

	for (int l = 0; l < grid.Nl; ++l) {
		for (int i = 0; i < grid.Nr; ++i) {
			double r = (i+1)*grid.dr;
			if (l == 0) {
				psi->data[i] = 2*r*exp(-r);
			} else {
				psi->data[i+l*grid.Nr] = 0.0;
			}
		}
	}

	double E(double t) {
		//return E0*(sin(omega*t) + 0.2*sin(2*omega*t + phi))*smoothpulse(t, T, 10*T);
		//return E0*sin(omega*t)*exp(-2*log(2)*pow(t-20*T, 2)/pow(tp, 2));
		return E0*cos(omega*(t-1.5*T))*exp(-pow(t-1.5*T, 2)/pow(tp, 2));
	}

	double t = 0.0;

	for (int i = 0; i < Nt; ++i) {
		double a = - E(t) - sphere_wavefunc_cos(psi, dUdz);
		fourier_update(P, a, Nt, dw, t, ws->dt);
		//printf("%.10e  %.10e\n", E(t), Eenv(t));

		sphere_kn_workspace_prop(ws, psi, E, t);

		t += ws->dt;
	}

	for (int i = 0; i < Nt; ++i) {
		printf("%f\n", cabs(P[i]));
	}

	sphere_kn_workspace_free(ws);
	sphere_wavefunc_free(psi);
	
	return 0;
}
