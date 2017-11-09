#pragma once

#include <complex.h>

#include "grid.h"

typedef double _Complex cdouble;

typedef double (*sh_f)(sh_grid_t const* grid, int ir, int il, int m);
