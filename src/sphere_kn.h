#pragma once

#include "fields.h"

#include "grid.h"
#include "sphere_wavefunc.h"
#include "ks_orbitals.h"

#include "utils.h"
#include "types.h"

typedef struct {
	double dt;

	sh_grid_t const* grid;

	sphere_pot_t U;
	sphere_pot_t Uabs;

	cdouble* b;
	cdouble* f;

	cdouble* alpha;
	cdouble* betta;
} sphere_kn_workspace_t;

sphere_kn_workspace_t*
sphere_kn_workspace_alloc(
		sh_grid_t const* grid,
		double const dt,
		sphere_pot_t U,
		sphere_pot_t Uabs);

void sphere_kn_workspace_free(sphere_kn_workspace_t* ws);

/* 
 * [1 + 0.5iΔtH(t+Δt/2)] Ψ(r, t+Δt) = [1 - 0.5iΔtH(t+Δt/2)] Ψ(r, t)
 * */

// exp(-0.5iΔtHang(l,m, t+Δt/2))
// @param E = E(t+dt/2)
void sphere_kn_workspace_prop_ang(
		sphere_kn_workspace_t* ws,
		sphere_wavefunc_t* wf,
		int l, double E);

// exp(-iΔtHat(l,m, t+Δt/2))
void sphere_kn_workspace_prop_at(
		sphere_kn_workspace_t* ws,
		sphere_wavefunc_t* wf,
		sphere_pot_t Ul,
		sphere_pot_t Uabs);

// O(dr^4)
void sphere_kn_workspace_prop_at_v2(
		sphere_kn_workspace_t* ws,
		sphere_wavefunc_t* wf,
		sphere_pot_t Ul,
		sphere_pot_t Uabs,
		bool img_time);

void sphere_kn_workspace_prop(
		sphere_kn_workspace_t* ws,
		sphere_wavefunc_t* wf,
		field_t field, double t);

void sphere_kn_workspace_prop_img(
		sphere_kn_workspace_t* ws,
		sphere_wavefunc_t* wf);

typedef struct {
	sphere_kn_workspace_t* wf_ws;
	double* Uh;
	double* Uxc;
	sp_grid_t* sp_grid;
} sphere_kn_orbs_workspace_t;

sphere_kn_orbs_workspace_t*
sphere_kn_orbs_workspace_alloc(
		sh_grid_t const* grid,
		double const dt,
		sphere_pot_t U,
		sphere_pot_t Uabs);

void sphere_kn_orbs_workspace_free(sphere_kn_orbs_workspace_t* ws);

void sphere_kn_orbs_workspace_prop_img(sphere_kn_orbs_workspace_t* ws, ks_orbitals_t* orbs);
