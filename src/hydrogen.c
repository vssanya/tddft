#include "hydrogen.h"

#include <math.h>

#include "utils.h"

double hydrogen_U(double r) {
	return -1.0/r;
}

double hydrogen_dUdz(double r) {
	return 1.0/pow(r, 2);
}

void hydrogen_ground(sphere_wavefunc_t* wf) {
	// l = 0
	double r = 0.0;
	for (int i = 0; i < wf->grid->Nr; ++i) {
		r += wf->grid->dr;
		wf->data[i] = 2*r*exp(-r);
	}

	for (int l = 1; l < wf->grid->Nl; ++l) {
		for (int i = 0; i < wf->grid->Nr; ++i) {
			wf->data[i+l*wf->grid->Nr] = 0.0;
		}
	}
}
