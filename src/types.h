#pragma once

#include <complex.h>

#include "grid.h"

typedef double complex cdouble;

typedef double (*sphere_pot_t)(sh_grid_t const* grid, int ir, int l, int m);
