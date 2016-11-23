#include "abs_pot.h"

#include "utils.h"


double Uabs(double r, sphere_grid_t const* grid) {
	double r_max = grid->dr*grid->Nr;
	double dr = 0.2*r_max;
	return 2*smoothstep(r, r_max-dr, r_max);
}
