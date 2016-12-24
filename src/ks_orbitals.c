#include "ks_orbitals.h"

ks_orbitals_t* ks_orbials_new(int ne, sh_grid_t const* grid) {
	ks_orbitals_t* orbs = malloc(sizeof(ks_orbitals_t));

	orbs->ne = ne;
	orbs->data = malloc(grid2_size(grid)*ne*sizeof(cdouble));

	orbs->wf = malloc(sizeof(sphere_wavefunc_t*)*ne);
	for (int ie = 0; ie < ne; ++ie) {
		orbs->wf[ie] = sphere_wavefunc_new_from(&orbs->data[grid2_size(grid)*ie], grid, 0);
	}
	
	return orbs;
}

void ks_orbitals_del(ks_orbitals_t* orbs) {
	for (int ie = 0; ie < orbs->ne; ++ie) {
		sphere_wavefunc_del(orbs->wf[ie]);
	}
	
	free(orbs->data);
	free(orbs);
}

double ks_orbitals_n(ks_orbitals_t const* orbs, int i[2]) {
	double res = 0.0;

	for (int ie = 0; ie < orbs->ne; ++ie) {
		cdouble const psi = swf_get_sp(orbs->wf[ie], (int[3]){i[0], i[1], 0});
		res += pow(creal(psi), 2) + pow(cimag(psi), 2);
	}

	return res;
}
