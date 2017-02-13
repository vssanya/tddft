#include "abs_pot.h"

#include "utils.h"


double Uabs(sh_grid_t const* grid, int ir, int il, int im) {
    double const r = sh_grid_r(grid, ir);
	double r_max = sh_grid_r_max(grid);
	double dr = 0.2*r_max;
	return 2*smoothstep(r, r_max-dr, r_max);
}

double uabs_zero(sh_grid_t const* grid, int ir, int il, int im) {
	return 0.0;
}
