#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "gps.h"
#include "common_alg.h"


ws_gps_t* ws_gps_alloc(sh_grid_t const* grid, atom_t const* atom, double dt, double e_max) {
	ws_gps_t* ws = malloc(sizeof(ws_gps_t));

	ws->grid = grid;
	ws->atom = atom;

	ws->dt = dt;
	ws->e_max = e_max;

	ws->s = NULL;

	ws->prop_wf = sh_wavefunc_new(grid, 0);

	return ws;
}

void ws_gps_free(ws_gps_t* ws) {
	if (ws->s != NULL) {
		free(ws->s);
	}
	sh_wavefunc_del(ws->prop_wf);
	free(ws);
}

void ws_gps_calc_s(ws_gps_t* ws, eigen_ws_t const* eigen) {
	int const Nr = ws->grid->n[iR];
	int const Nl = ws->grid->n[iL];

	int const Ne = eigen_get_n_with_energy(eigen, ws->e_max);

	ws->s = calloc(Nl*Nr*Nr, sizeof(cdouble));

#pragma omp parallel for collapse(3)
	for (int il = 0; il < Nl; ++il) {
		for (int ir1 = 0; ir1 < Nr; ++ir1) {
			for (int ir2 = 0; ir2 < Nr; ++ir2) {
				for (int ie = 0; ie < Ne; ++ie) {
					ws->s[ir2 + Nr*(ir1 + il*Nr)] += cexp(-I*ws->dt*eigen_eval(eigen, il, ie))*eigen_evec(eigen, il, ir1, ie)*eigen_evec(eigen, il, ir2, ie);
				}
			}
		}
	}
}

void ws_gps_prop(ws_gps_t const* ws, sh_wavefunc_t* wf) {
	int const Nr = ws->grid->n[iR];
	int const Nl = ws->grid->n[iL];

#pragma omp parallel for
	for (int il = 0; il < Nl; ++il) {
		cdouble* psi = swf_ptr(ws->prop_wf, 0, il);

		for (int ir1 = 0; ir1 < Nr; ++ir1) {
			psi[ir1] = 0.0;
			for (int ir2 = 0; ir2 < Nr; ++ir2) {
				psi[ir1] += ws->s[ir2 + (ir1 + il*Nr)*Nr]*swf_get(wf, ir2, il);
			}
		}
	}

	sh_wavefunc_copy(ws->prop_wf, wf);
}

void ws_gps_prop_common(
		ws_gps_t* ws,
		sh_wavefunc_t* wf,
		uabs_sh_t const* uabs,
		field_t const* field,
		double t
) {
	double Et = field_E(field, t + ws->dt/2);

	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return r*Et*clm(l,m);
	}

	int l1 = 1;
	for (int il = 0; il < ws->grid->n[iL] - l1; ++il) {
		wf_prop_ang_l(wf, ws->dt*0.5, il, l1, Ul1);
	}

	ws_gps_prop(ws, wf);

	for (int il = ws->grid->n[iL] - 1 - l1; il >= 0; --il) {
		wf_prop_ang_l(wf, ws->dt*0.5, il, l1, Ul1);
	}

	for (int il = 0; il < ws->grid->n[iL]; ++il) {
		for (int ir = 0; ir < ws->grid->n[iR]; ++ir) {
			wf->data[ir + il*ws->grid->n[iR]]*=exp(-uabs_get(uabs, ws->grid, ir, il, wf->m)*ws->dt);
		}
	}
}