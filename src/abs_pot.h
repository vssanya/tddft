#pragma once

#include "grid.h"

#ifdef __cplusplus
extern "C" {
#endif

double uabs(ShGrid const* grid, int ir, int il, int im) __attribute__((pure));

__attribute__((pure))
double mask_core(ShGrid const* grid, int ir, int il, int im);

typedef double (*uabs_func_t)(void const* self, ShGrid const* grid, int ir, int il, int im);
typedef struct uabs_sh_s {
	uabs_func_t func;
} uabs_sh_t;

double uabs_get(uabs_sh_t const* self, ShGrid const* grid, int ir, int il, int im);

typedef struct {
	uabs_func_t func;
	double l[3];
	double u[3];
} uabs_multi_hump_t;

uabs_multi_hump_t* uabs_multi_hump_new(double lambda_min, double lambda_max);
double uabs_multi_hump_func(uabs_multi_hump_t const* self, ShGrid const* grid, int ir, int il, int im);

double uabs_zero_func(uabs_sh_t const* uabs, ShGrid const* grid, int ir, int il, int im);
static const uabs_sh_t uabs_zero = {.func = (uabs_func_t)uabs_zero_func};

#ifdef __cplusplus
}
#endif
