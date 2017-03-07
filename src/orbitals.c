#include "orbitals.h"

orbitals_t* ks_orbials_new(int ne, sh_grid_t const* grid) {
	orbitals_t* orbs = malloc(sizeof(orbitals_t));

	orbs->ne = ne;
	orbs->grid = grid;
	orbs->data = malloc(grid2_size(grid)*ne*sizeof(cdouble));

	orbs->wf = malloc(sizeof(sh_wavefunc_t*)*ne);
	for (int ie = 0; ie < ne; ++ie) {
		orbs->wf[ie] = sh_wavefunc_new_from(&orbs->data[grid2_size(grid)*ie], grid, 0);
	}
	
	return orbs;
}

void orbitals_del(orbitals_t* orbs) {
	for (int ie = 0; ie < orbs->ne; ++ie) {
		sh_wavefunc_del(orbs->wf[ie]);
	}
	
	free(orbs->data);
	free(orbs);
}

double orbitals_n(orbitals_t const* orbs, sp_grid_t const* grid, int i[2]) {
	double res = 0.0;

	for (int ie = 0; ie < orbs->ne; ++ie) {
		cdouble const psi = swf_get_sp(orbs->wf[ie], grid, (int[3]){i[0], i[1], 0});
		res += pow(creal(psi), 2) + pow(cimag(psi), 2);
	}

	return 2.0*res;
}

double orbitals_norm(orbitals_t const* orbs) {
	double res = 0.0;
#pragma omp target map(tofrom: res)
#pragma omp parallel for reduction(+:res)
	for (int ie=0; ie<orbs->ne; ++ie) {
		res += sh_wavefunc_norm(orbs->wf[ie]);
	}
	return 2*res;
}

void orbitals_normalize(orbitals_t* orbs) {
	for (int ie=0; ie<orbs->ne; ++ie) {
		sh_wavefunc_normalize(orbs->wf[ie]);
	}
}

double orbitals_cos(orbitals_t const* orbs, sh_f U) {
	double res = 0.0;
#pragma omp target map(tofrom: res)
#pragma omp parallel for reduction(+:res)
	for (int ie=0; ie<orbs->ne; ++ie) {
		res += sh_wavefunc_cos(orbs->wf[ie], U);
	}
	return 2*res;
}
