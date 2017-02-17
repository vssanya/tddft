#include "argon.h"

void argon_init(ks_orbitals_t* orbs) {
	assert(orbs->ne == 9);

	// init s states
	for (int ie = 0; ie < 3; ++ie) {
		sphere_wavefunc_random_l(orbs->wf[ie], 0);
	}

	// init p states
	for (int ie = 0; ie < 3*2; ++ie) {
		sphere_wavefunc_random_l(orbs->wf[ie+3], 1);
	}

	orbs->wf[0]->m = 0;
	orbs->wf[1]->m = 0;
	orbs->wf[2]->m = 0;

	orbs->wf[3]->m = -1;
	orbs->wf[4]->m = -1;
	orbs->wf[5]->m = 0;
	orbs->wf[6]->m = 0;
	orbs->wf[7]->m = 1;
	orbs->wf[8]->m = 1;
}

void argon_ort(ks_orbitals_t* orbs) {
	sphere_wavefunc_ort_l(0, 3, orbs->wf);

	sphere_wavefunc_ort_l(1, 2, &orbs->wf[3]);
	sphere_wavefunc_ort_l(1, 2, &orbs->wf[5]);
	sphere_wavefunc_ort_l(1, 2, &orbs->wf[7]);
}

double argon_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
    double const r = sh_grid_r(grid, ir);
	return -18.0/r;
}

double argon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
    double const r = sh_grid_r(grid, ir);
    return 18.0/pow(r, 2);
}
