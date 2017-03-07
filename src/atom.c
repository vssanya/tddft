#include "atom.h"

void atom_argon_init(orbitals_t* orbs) {
	assert(orbs->ne == 9);

	// init s states
	for (int ie = 0; ie < 3; ++ie) {
		sh_wavefunc_random_l(orbs->wf[ie], 0);
	}

	// init p states
	for (int ie = 0; ie < 3*2; ++ie) {
		sh_wavefunc_random_l(orbs->wf[ie+3], 1);
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

void atom_neon_init(orbitals_t* orbs) {
	assert(orbs->ne == 5);

	// init s states
	for (int ie = 0; ie < 2; ++ie) {
		sh_wavefunc_random_l(orbs->wf[ie], 0);
	}

	// init p states
	for (int ie = 0; ie < 3; ++ie) {
		sh_wavefunc_random_l(orbs->wf[ie+2], 1);
	}

	orbs->wf[0]->m = 0;
	orbs->wf[1]->m = 0;

	orbs->wf[2]->m = -1;
	orbs->wf[3]->m = 0;
	orbs->wf[4]->m = 1;
}

void atom_hydrogen_init(orbitals_t* orbs) {
	assert(orbs->ne == 1);

	atom_hydrogen_ground(orbs->wf[0]);
}


void atom_argon_ort(orbitals_t* orbs) {
	sh_wavefunc_ort_l(0, 3, orbs->wf);

	sh_wavefunc_ort_l(1, 2, &orbs->wf[3]);
	sh_wavefunc_ort_l(1, 2, &orbs->wf[5]);
	sh_wavefunc_ort_l(1, 2, &orbs->wf[7]);
}

void atom_neon_ort(orbitals_t* orbs) {
	sh_wavefunc_ort_l(0, 2, orbs->wf);
}

double atom_argon_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
    double const r = sh_grid_r(grid, ir);
	return -18.0/r;
}

double atom_argon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
    double const r = sh_grid_r(grid, ir);
    return 18.0/pow(r, 2);
}

double atom_neon_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
    double const r = sh_grid_r(grid, ir);
	return -10.0/r;
}

double atom_neon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
    double const r = sh_grid_r(grid, ir);
    return 10.0/pow(r, 2);
}

double atom_hydrogen_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
    double const r = sh_grid_r(grid, ir);
	return -1.0/r;
}

double atom_hydrogen_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
    double const r = sh_grid_r(grid, ir);
    return 1.0/pow(r, 2);
}

void atom_hydrogen_ground(sh_wavefunc_t* wf) {
	assert(wf->m == 0);
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
