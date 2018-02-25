#pragma once

#include <math.h>
#include "../grid.h"


void integrate_rmin_rmax_o5(int n, ShGrid const* grid, double const* f, double* res);
void integrate_rmin_rmax_o3(int n, ShGrid const* grid, double const* f, double* res);
