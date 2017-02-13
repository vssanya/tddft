#include "hydrogen.h"

#include <math.h>

#include "utils.h"

double hydrogen_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
    double const r = sh_grid_r(grid, ir);
	return -1.0/r;
}

double hydrogen_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
    double const r = sh_grid_r(grid, ir);
    return 1.0/pow(r, 2);
}

void hydrogen_ground(sphere_wavefunc_t* wf) {
	// l = 0
	{
		int const il = 0;
		for (int ir = 0; ir < wf->grid->n[iR]; ++ir) {
			double r = sh_grid_r(wf->grid, ir);
			swf_set(wf, ir, il, 2*r*exp(-r));
		}
	}

	for (int il = 1; il < wf->grid->n[iL]; ++il) {
		for (int ir = 0; ir < wf->grid->n[iR]; ++ir) {
			swf_set(wf, ir, il, 0.0);
		}
	}
}
