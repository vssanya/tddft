#include "sh_wavefunc.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <omp.h>

#include "utils.h"

sh_wavefunc_t* _sh_wavefunc_new(cdouble* data, bool data_own, sh_grid_t const* grid, int const m) {
	sh_wavefunc_t* wf = malloc(sizeof(sh_wavefunc_t));
	wf->grid = grid;

	wf->data = data;
	wf->data_own = data_own;

	wf->m = m;

	return wf;
}

sh_wavefunc_t* sh_wavefunc_new(sh_grid_t const* grid, int const m) {
	cdouble* data = calloc(grid2_size(grid), sizeof(cdouble));
	return _sh_wavefunc_new(data, true, grid, m);
}

sh_wavefunc_t* sh_wavefunc_new_from(cdouble* data, sh_grid_t const* grid, int const m) {
	return _sh_wavefunc_new(data, false, grid, m);
}

void sh_wavefunc_del(sh_wavefunc_t* wf) {
	if (wf->data_own) {
		free(wf->data);
	}
	free(wf);
}

cdouble sh_wavefunc_prod(sh_wavefunc_t const* wf1, sh_wavefunc_t const* wf2) {
	cdouble res = 0.0;
	for (int i = 0; i < grid2_size(wf1->grid); ++i) {
		res += wf2->data[i]*conj(wf1->data[i]);
	}
	return res*wf1->grid->d[iR];
}

void sh_wavefunc_ort_l(int l, int n, sh_wavefunc_t* wfs[n]) {
	assert(n > 1);
	for (int in=1; in<n; ++in) {
		assert(wfs[in-1]->m == wfs[in]->m);
	}

	sh_grid_t const* grid = wfs[0]->grid;

	cdouble proj[n];
	cdouble norm[n];

	for (int in=0; in<n; ++in) {
		for (int ip=0; ip<in; ++ip) {
			proj[ip] = sh_wavefunc_prod(wfs[ip], wfs[in]) / norm[ip];
		}

		for (int ip=0; ip<in; ++ip) {
			cdouble* psi = swf_ptr(wfs[in], 0, l);
			for (int ir=0; ir<grid->n[iR]; ++ir) {
				psi[ir] -= proj[ip]*swf_get(wfs[ip], ir, l);
			}
		}

		norm[in] = sh_wavefunc_norm(wfs[in]);
	}
}

void sh_wavefunc_n_sp(sh_wavefunc_t const* wf, sp_grid_t const* grid, double n[grid->n[iR]*grid->n[iC]], ylm_cache_t const* ylm_cache) {
#pragma omp parallel for collapse(2)
	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		for (int ic = 0; ic < grid->n[iC]; ++ic) {
			cdouble const psi = swf_get_sp(wf, grid, (int[3]){ir, ic, 0}, ylm_cache);
			n[ir + ic*grid->n[iR]] = pow(creal(psi), 2) + pow(cimag(psi), 2);
		}
	}
}

double sh_wavefunc_norm(sh_wavefunc_t const* wf) {
	double norm = 0.0;

	for (int i = 0; i < grid2_size(wf->grid); ++i) {
		cdouble value = wf->data[i];
		norm += pow(creal(value), 2) + pow(cimag(value), 2);
	}
	return norm*wf->grid->d[iR];
}

void sh_wavefunc_normalize(sh_wavefunc_t* wf) {
	double norm = sh_wavefunc_norm(wf);
	for (int i = 0; i < grid2_size(wf->grid); ++i) {
		wf->data[i] /= sqrt(norm);
	}
}

void sh_wavefunc_print(sh_wavefunc_t const* wf) {
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
double sh_wavefunc_cos(sh_wavefunc_t const* wf, sh_f U) {
	double res = 0.0;

	for (int il = 0; il < wf->grid->n[iL]-1; ++il) {
		double res_l = 0.0;
		for (int ir = 0; ir < wf->grid->n[iR]; ++ir) {
            res_l += creal(swf_get(wf, ir, il)*conj(swf_get(wf, ir, il+1)))*U(wf->grid, ir, il, wf->m);
		}
		int const l = sh_grid_l(wf->grid, il);
		res += res_l*clm(l, wf->m);
	}

	res *= 2*wf->grid->d[iR];
	return res;
}

double sh_wavefunc_z(sh_wavefunc_t const* wf) {
    double func(sh_grid_t const* grid, int ir, int il, int im) { return sh_grid_r(grid, ir); }

	return sh_wavefunc_cos(wf, func);
}

cdouble swf_get_sp(sh_wavefunc_t const* wf, sp_grid_t const* grid, int i[3], ylm_cache_t const* ylm_cache) {
	cdouble res = 0.0;
	double r = sh_grid_r(wf->grid, i[iR]);
	for (int il = 0; il < wf->grid->n[iL]; ++il) {
		int const l = sh_grid_l(wf->grid, il);
		res += swf_get(wf, i[iR], il)*ylm_cache_get(ylm_cache, l, wf->m, i[iC]) / r;
	}
	return res;
}

void sh_wavefunc_random_l(sh_wavefunc_t* wf, int l) {
	assert(l >= 0 && l < wf->grid->n[iL]);

	for (int il=0; il<l; ++il) {
		for (int ir=0; ir<wf->grid->n[iR]; ++ir) {
			swf_set(wf, ir, il, 0.0);
		}
	}

	{
		int il = l;
		for (int ir=0; ir<wf->grid->n[iR]; ++ir) {
			double const r = sh_grid_r(wf->grid, ir);
			swf_set(wf, ir, il, (double)rand()/(double)RAND_MAX*exp(-r/(10*l+1)));
		}
	}

	for (int il=l+1; il<wf->grid->n[iL]; ++il) {
		for (int ir=0; ir<wf->grid->n[iR]; ++ir) {
			swf_set(wf, ir, il, 0.0);
		}
	}
}
