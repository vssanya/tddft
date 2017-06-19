#pragma once

#include "fields.h"

#include "grid.h"
#include "sh_wavefunc.h"
#include "orbitals.h"
#include "abs_pot.h"
#include "atom.h"
#include "eigen.h"
#include "hartree_potential.h"

#include "utils.h"
#include "types.h"

typedef struct {
	sh_grid_t const* grid;
	atom_t const* atom;

	double dt;
	double e_max; // maximum energy

	cdouble* s; // propogation matrix shape = (Nl,Nr,n_evec)
	int n_evec; // number of eigenvec for prop

	sh_wavefunc_t* prop_wf;
} gps_ws_t;

gps_ws_t* gps_ws_alloc(sh_grid_t const* grid, atom_t const* atom, double dt, double e_max);
void gps_ws_free(gps_ws_t* ws);
void gps_ws_calc_s(gps_ws_t* ws, eigen_ws_t const* eigen);
void gps_ws_prop(gps_ws_t const* ws, sh_wavefunc_t* wf);
void gps_ws_prop_common(
		gps_ws_t* ws,
		sh_wavefunc_t* wf,
		uabs_sh_t const* uabs,
		field_t const* field,
		double t
);

/*! \file
 * Split-step method:
 * \f[ e^{(A + B)dt} = e^\frac{Adt}{2} e^{Bdt} e^\frac{Adt}{2} + \frac{1}{24}\left[A + 2B, [A,B]\right] dt^3 + O(dt^4) \f]
 *
 * \f[ C_1 = \left[\frac{d^2}{dr^2}, r\right] = 2\frac{d}{dr} \f]
 * \f[ C_2 = \left[\frac{d^2}{dr^2} + 2r, C_1\right] = [2r, C_1] = 2(\frac{d}{dr} - 1) \f]
 *
 * For \f$A = 1/2 d^2/dr^2\f$ and \f$B = r\cos{\theta}E\f$:
 * 
 * */

typedef struct {
	sh_grid_t const* grid;
	uabs_sh_t const* uabs;

	cdouble* alpha;
	cdouble* betta;

	int num_threads;
} sh_workspace_t;

sh_workspace_t*
sh_workspace_alloc(
		sh_grid_t const* grid,
		uabs_sh_t const* uabs,
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

// O(dr^4)
void sh_workspace_prop_at(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		cdouble dt,
		sh_f Ul,
		int Z,
    potential_type_e u_type
);

void sh_workspace_prop(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		atom_t const* atom,
		field_t const* field,
		double t,
		double dt
);

void sh_workspace_prop_img(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		atom_t const* atom,
		double dt
);

typedef struct {
	sh_workspace_t* wf_ws;

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
} sh_orbs_workspace_t;

sh_orbs_workspace_t*
sh_orbs_workspace_alloc(
		sh_grid_t const* sh_grid,
		sp_grid_t const* sp_grid,
		uabs_sh_t const* uabs,
		ylm_cache_t const* ylm_cache,
		int Uh_lmax,
		int Uxc_lmax,
		potential_xc_f Uxc,
		int num_threads
);

void sh_orbs_workspace_free(sh_orbs_workspace_t* ws);
void sh_orbs_workspace_prop(
		sh_orbs_workspace_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		field_t const* field,
		double t,
		double dt
);
void sh_orbs_workspace_prop_img(
		sh_orbs_workspace_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		double dt
);

void prop_ang_l(sh_wavefunc_t* wf, cdouble dt, int l, int l1, sh_f Ul);
