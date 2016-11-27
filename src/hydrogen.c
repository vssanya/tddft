#include "hydrogen.h"

#include <math.h>

#include "utils.h"

double hydrogen_U(double r) {
	return -1.0/r;
}

double hydrogen_dUdz(double r) {
	return 1.0/pow(r, 2);
}

sphere_wavefunc_t* hydrogen_ground(sphere_grid_t const* grid) {
	sphere_wavefunc_t* wf = sphere_wavefunc_alloc(grid, 0);

	// l = 0
	double r = 0.0;
	for (int i = 0; i < grid->Nr; ++i) {
		r += grid->dr;
		wf->data[i] = 2*r*exp(-r);
	}

	for (int l = 1; l < grid->Nl; ++l) {
		for (int i = 0; i < grid->Nr; ++i) {
			wf->data[i+l*grid->Nr] = 0.0;
		}
	}

	return wf;
}
