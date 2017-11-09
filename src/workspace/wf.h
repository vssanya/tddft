#pragma once

#include "../fields.h"

#include "../grid.h"
#include "../sh_wavefunc.h"
#include "../orbitals.h"
#include "../abs_pot.h"
#include "../atom.h"

#include "../utils.h"
#include "../types.h"

#ifdef __cplusplus
extern "C" {
#endif

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
} ws_wf_t;

ws_wf_t* ws_wf_new(sh_grid_t const* grid, uabs_sh_t const* uabs, int num_threads);
void ws_wf_del(ws_wf_t* ws);

/* 
 * [1 + 0.5iΔtH(t+Δt/2)] Ψ(r, t+Δt) = [1 - 0.5iΔtH(t+Δt/2)] Ψ(r, t)
 * */

// exp(-0.5iΔtHang(l,m, t+Δt/2))
// @param E = E(t+dt/2)
void ws_wf_prop_ang(
		ws_wf_t* ws,
		sh_wavefunc_t* wf,
		double dt,
		int l, double E
		);

// O(dr^4)
void ws_wf_prop_at(
		ws_wf_t* ws,
		sh_wavefunc_t* wf,
		cdouble dt,
		sh_f Ul,
		int Z,
		potential_type_e u_type
		);

void ws_wf_prop(
		ws_wf_t* ws,
		sh_wavefunc_t* wf,
		atom_t const* atom,
		field_t const* field,
		double t,
		double dt
		);

void ws_wf_prop_img(
		ws_wf_t* ws,
		sh_wavefunc_t* wf,
		atom_t const* atom,
		double dt
		);

void ws_wf_prop_common(
		ws_wf_t* ws,
		sh_wavefunc_t* wf,
		cdouble dt,
		int l_max,
		sh_f Ul[l_max],
		uabs_sh_t const* uabs,
		int Z,
		potential_type_e u_type
		);

#ifdef __cplusplus
}
#endif
