#pragma once

#include <complex.h>

#include "sphere_grid.h"

typedef double complex cdouble;

typedef double (*sphere_pot_t)(double r);
typedef double (*sphere_pot_abs_t)(double r, sphere_grid_t const* grid);
