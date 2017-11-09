#pragma once

#include "../fields.h"

#include "../grid.h"
#include "../sh_wavefunc.h"
#include "../orbitals.h"
#include "../abs_pot.h"
#include "../atom.h"
#include "../eigen.h"

#include "../utils.h"
#include "../types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	sh_grid_t const* grid;
	atom_t const* atom;

	double dt;
	double e_max; // maximum energy

	cdouble* s; // propogation matrix shape = (Nl,Nr,n_evec)
	int n_evec; // number of eigenvec for prop

	sh_wavefunc_t* prop_wf;
} ws_gps_t;

ws_gps_t* ws_gps_alloc(sh_grid_t const* grid, atom_t const* atom, double dt, double e_max);
void ws_gps_free(ws_gps_t* ws);
void ws_gps_calc_s(ws_gps_t* ws, eigen_ws_t const* eigen);
void ws_gps_prop(ws_gps_t const* ws, sh_wavefunc_t* wf);
void ws_gps_prop_common(
		ws_gps_t* ws,
		sh_wavefunc_t* wf,
		uabs_sh_t const* uabs,
		field_t const* field,
		double t
);

#ifdef __cplusplus
}
#endif
