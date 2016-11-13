#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "sphere_kn.h"

const double M_PI = 3.14;

double U(double r) {
	return -1.0/r;
}

int main(int argc, char *argv[])
{
	sphere_grid_t grid;
	grid.Nr = 1000;
	grid.Nl = 2;
	grid.dr = 0.2;

	double dt = 0.1;

	sphere_kn_workspace_t* ws = sphere_kn_workspace_alloc(&grid, dt, U);
	sphere_wavefunc_t* psi = sphere_wavefunc_alloc(&grid, 0);

	for (int i = 0; i < grid.Nr; ++i) {
		double r = (i+1)*grid.dr;
		psi->data[i]         = 2*r*exp(-r) + 0.5;
		psi->data[i+grid.Nr] = 0.0;
	}

	for (int i = 0; i<400; ++i) {
		if (i%10 == 0) {
			sphere_wavefunc_print(psi, 0);
		}

		sphere_kn_workspace_prop(ws, psi);
		sphere_wavefunc_normalize(psi);
	}

	sphere_kn_workspace_free(ws);
	sphere_wavefunc_free(psi);
	
	return 0;
}
