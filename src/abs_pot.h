#pragma once

#include "grid.h"

double uabs(sh_grid_t const* grid, int ir, int il, int im) __attribute__((pure));

__attribute__((pure))
double mask_core(sh_grid_t const* grid, int ir, int il, int im);

typedef struct uabs_sh_s {
	double (*func)(void const* self, sh_grid_t const* grid, int ir, int il, int im);
} uabs_sh_t;

double uabs_get(uabs_sh_t const* self, sh_grid_t const* grid, int ir, int il, int im);

typedef struct {
	double (*func)(void const* self, sh_grid_t const* grid, int ir, int il, int im);
	double l[3];
	double u[3];
} uabs_multi_hump_t;

uabs_multi_hump_t* uabs_multi_hump_new(double lambda_min, double lambda_max);
double uabs_multi_hump_func(uabs_multi_hump_t const* self, sh_grid_t const* grid, int ir, int il, int im);

double uabs_zero_func(uabs_sh_t const* uabs, sh_grid_t const* grid, int ir, int il, int im);
static const uabs_sh_t uabs_zero = {.func = uabs_zero_func};
