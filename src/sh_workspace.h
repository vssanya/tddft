#pragma once

#include "fields.h"

#include "grid.h"
#include "sh_wavefunc.h"
#include "orbitals.h"

#include "utils.h"
#include "types.h"

typedef struct {
	sh_grid_t const* grid;

	sh_f U;
	sh_f Uabs;

	cdouble* b;
	cdouble* f;

	cdouble* alpha;
	cdouble* betta;

	int num_threads;
} sh_workspace_t;

sh_workspace_t*
sh_workspace_alloc(
		sh_grid_t const* grid,
		sh_f U,
		sh_f Uabs,
		int num_threads);

void sh_workspace_free(sh_workspace_t* ws);

/* 
 * [1 + 0.5iΔtH(t+Δt/2)] Ψ(r, t+Δt) = [1 - 0.5iΔtH(t+Δt/2)] Ψ(r, t)
 * */

// exp(-0.5iΔtHang(l,m, t+Δt/2))
// @param E = E(t+dt/2)
void sh_workspace_prop_ang(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		double dt,
		int l, double E);

// exp(-iΔtHat(l,m, t+Δt/2))
void sh_workspace_prop_at(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		double dt,
		sh_f Ul,
		sh_f Uabs);

// O(dr^4)
void sh_workspace_prop_at_v2(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		cdouble dt,
		sh_f Ul,
		sh_f Uabs
);

void sh_workspace_prop(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		field_t field,
		double t,
		double dt
);

void sh_workspace_prop_img(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		double dt
);

typedef struct {
	sh_workspace_t** wf_ws;
	double* Uh;
	double* Uxc;
	sh_grid_t const* sh_grid;
	sp_grid_t* sp_grid;
	int num_threads;
	double* uh_tmp;
} sh_orbs_workspace_t;

sh_orbs_workspace_t*
sh_orbs_workspace_alloc(
		sh_grid_t const* grid,
		sh_f U,
		sh_f Uabs,
		int num_threads
);

void sh_orbs_workspace_free(sh_orbs_workspace_t* ws);
void sh_orbs_workspace_prop(
		sh_orbs_workspace_t* ws,
		orbitals_t* orbs,
		field_t field,
		double t,
		double dt
);
void sh_orbs_workspace_prop_img(
		sh_orbs_workspace_t* ws,
		orbitals_t* orbs,
		double dt
);
