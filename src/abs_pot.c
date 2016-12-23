#include "abs_pot.h"

#include "utils.h"


double Uabs(double r, sh_grid_t const* grid) {
	double r_max = sh_grid_r_max(grid);
	double dr = 0.2*r_max;
	return 2*smoothstep(r, r_max-dr, r_max);
}
