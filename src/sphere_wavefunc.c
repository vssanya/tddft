#include "sphere_wavefunc.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utils.h"

sphere_wavefunc_t* sphere_wavefunc_new(sh_grid_t const* grid, int const m) {
	sphere_wavefunc_t* wf = malloc(sizeof(sphere_wavefunc_t));
	wf->grid = grid;
	wf->data = calloc(grid2_size(grid), sizeof(cdouble));
	wf->m = m;

	return wf;
}

void sphere_wavefunc_del(sphere_wavefunc_t* wf) {
	free(wf->data);
	free(wf);
}

double sphere_wavefunc_norm(sphere_wavefunc_t const* wf) {
	double norm = 0.0;
	for (int i = 0; i < grid2_size(wf->grid); ++i) {
		norm += wf->data[i]*conj(wf->data[i]);
	}
	return norm*wf->grid->d[iR];
}

void sphere_wavefunc_normalize(sphere_wavefunc_t* wf) {
	double norm = sphere_wavefunc_norm(wf);
	for (int i = 0; i < grid2_size(wf->grid); ++i) {
		wf->data[i] /= sqrt(norm);
	}
}

void sphere_wavefunc_print(sphere_wavefunc_t const* wf) {
	for (int ir = 0; ir < wf->grid->n[iR]; ++ir) {
		double const r = sh_grid_r(wf->grid, ir);
		double res = 0.0;
		for (int il = 0; il < wf->grid->n[iL]; ++il) {
			res += pow(cabs(swf_get(wf, ir, il)), 2);
		}
		printf("%f ", res/(r*r));
	}
	printf("\n");
}

// <psi|U(r)cos(\theta)|psi>
double sphere_wavefunc_cos(sphere_wavefunc_t const* wf, sphere_pot_t U) {
	double res = 0.0;

	for (int il = 0; il < wf->grid->n[iL]-1; ++il) {
		double res_l = 0.0;
		for (int ir = 0; ir < wf->grid->n[iR]; ++ir) {
			double const r = sh_grid_r(wf->grid, ir);
			res_l += creal(swf_get(wf, ir, il)*conj(swf_get(wf, ir, il+1)))*U(r);
		}
		int const l = sh_grid_l(wf->grid, il);
		res += res_l*clm(l, wf->m);
	}

	res *= 2*wf->grid->d[iR];
	return res;
}

double sphere_wavefunc_z(sphere_wavefunc_t const* wf) {
	double func(double r) { return r; }

	return sphere_wavefunc_cos(wf, func);
}

cdouble swf_get_sp(sphere_wavefunc_t const* wf, int i[3]) {
	cdouble res = 0.0;
	double r = sh_grid_r(wf->grid, i[iR]);
	for (int il = 0; il < wf->grid->n[iL]; ++il) {
		int const l = sh_grid_l(wf->grid, il);
		res += swf_get(wf, i[iR], il)*Ylm(l, wf->m, i[iC]) / r;
	}
	return res;
}
