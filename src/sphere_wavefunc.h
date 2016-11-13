#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "types.h"
#include "sphere_grid.h"

typedef struct {
	sphere_grid_t const* grid;
	int m;
	cdouble* data; // data[ix][il] = data[ix + il*Nr]
} sphere_wavefunc_t;

sphere_wavefunc_t* sphere_wavefunc_alloc(sphere_grid_t const* grid, int const m) {
	sphere_wavefunc_t* wf = malloc(sizeof(sphere_wavefunc_t));
	wf->grid = grid;
	wf->data = malloc(sizeof(cdouble)*grid->Nr*grid->Nl);
	wf->m = m;

	return wf;
}

void sphere_wavefunc_free(sphere_wavefunc_t* wf) {
	free(wf->data);
	free(wf);
}

double sphere_wavefunc_norm(sphere_wavefunc_t const* wf) {
	double norm = 0.0;
	for (int i = 0; i < wf->grid->Nl*wf->grid->Nr; ++i) {
		norm += wf->data[i]*conj(wf->data[i]);
	}
	return norm*wf->grid->dr;
}

void sphere_wavefunc_normalize(sphere_wavefunc_t* wf) {
	double norm = sphere_wavefunc_norm(wf);
	for (int i = 0; i < wf->grid->Nl*wf->grid->Nr; ++i) {
		wf->data[i] /= sqrt(norm);
	}
}

void sphere_wavefunc_print(sphere_wavefunc_t const* wf, int const l) {
	for (int i = 0; i < wf->grid->Nr; ++i) {
		printf("%f ", creal(wf->data[i + l*wf->grid->Nr]));
	}
	printf("\n");
}
