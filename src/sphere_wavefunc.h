#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "types.h"
#include "sphere_grid.h"
#include "utils.h"

typedef struct {
	sphere_grid_t const* grid;
	int m;
	cdouble* data; // data[ix][il] = data[ix + il*Nr]
} sphere_wavefunc_t;

sphere_wavefunc_t* sphere_wavefunc_alloc(sphere_grid_t const* grid, int const m) {
	sphere_wavefunc_t* wf = malloc(sizeof(sphere_wavefunc_t));
	wf->grid = grid;
	wf->data = calloc(grid->Nr*grid->Nl, sizeof(cdouble));
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

void sphere_wavefunc_print(sphere_wavefunc_t const* wf) {
	for (int i = 0; i < wf->grid->Nr; ++i) {
		double res = 0.0;
		for (int l = 0; l < wf->grid->Nl; ++l) {
			res += pow(cabs(wf->data[i + l*wf->grid->Nr]), 2);
		}
		printf("%f ", res);
	}
	printf("\n");
}

// <psi|U(r)cos(\theta)|psi>
double sphere_wavefunc_cos(sphere_wavefunc_t const* wf, sphere_pot_t U) {
	int const Nr = wf->grid->Nr;
	double res = 0.0;

	for (int l = 0; l < wf->grid->Nl-1; ++l) {
		double r = 0.0;
		double res_l = 0.0;
		for (int i = 0; i < Nr; ++i) {
			r += wf->grid->dr;
			res_l += creal(wf->data[i + l*Nr]*conj(wf->data[i + (l+1)*Nr]))*U(r);
		}
		res += res_l*clm(l, wf->m);
	}

	res *= 2*wf->grid->dr;
	return res;
}
