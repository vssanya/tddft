#pragma once

#include "grid.h"

double Uabs(sh_grid_t const* grid, int ir, int il, int im) __attribute__((pure));
double uabs_zero(sh_grid_t const* grid, int ir, int il, int im) __attribute__((pure));
