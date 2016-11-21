#pragma once

#include "types.h"
#include "sphere_grid.h"

typedef struct {
	sphere_grid_t const* grid;
	int m;
	cdouble* data; // data[ix][il] = data[ix + il*Nr]
} sphere_wavefunc_t;

sphere_wavefunc_t* sphere_wavefunc_alloc(
		sphere_grid_t const* grid,
		int const m
);

void   sphere_wavefunc_free(sphere_wavefunc_t* wf);

double sphere_wavefunc_norm(sphere_wavefunc_t const* wf);

void   sphere_wavefunc_normalize(sphere_wavefunc_t* wf);

void   sphere_wavefunc_print(sphere_wavefunc_t const* wf);

// <psi|U(r)cos(\theta)|psi>
double sphere_wavefunc_cos(
		sphere_wavefunc_t const* wf,
		sphere_pot_t U
);

double sphere_wavefunc_z(sphere_wavefunc_t const* wf);
