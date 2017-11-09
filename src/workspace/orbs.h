#pragma once

#include "wf.h"
#include "../orbitals.h"

#include "../hartree_potential.h"
#include "../utils.h"
#include "../types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ws_wf_t* wf_ws;

	double* Utmp;
	double* Utmp_local;

	int Uh_lmax;

	potential_xc_f Uxc;
	int Uxc_lmax;

	double* Uee;
	int lmax;

	sh_grid_t const* sh_grid;
	sp_grid_t const* sp_grid;
	double* uh_tmp;
	double* n_sp; // for root
	double* n_sp_local;
	ylm_cache_t const* ylm_cache;
} ws_orbs_t;

ws_orbs_t*
ws_orbs_alloc(
		sh_grid_t const* sh_grid,
		sp_grid_t const* sp_grid,
		uabs_sh_t const* uabs,
		ylm_cache_t const* ylm_cache,
		int Uh_lmax,
		int Uxc_lmax,
		potential_xc_f Uxc,
		int num_threads
);

void ws_orbs_free(ws_orbs_t* ws);
void ws_orbs_prop(
		ws_orbs_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		field_t const* field,
		double t,
		double dt,
		bool calc_uee
);
void ws_orbs_prop_img(
		ws_orbs_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		double dt
);

void ws_orbs_calc_Uee(ws_orbs_t* ws, orbitals_t const* orbs, int Uxc_lmax, int Uh_lmax);

#ifdef __cplusplus
}
#endif
