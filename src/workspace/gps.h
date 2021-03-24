#pragma once

#include "../fields.h"

#include "../grid.h"
#include "../wavefunc/sh_2d.h"
#include "../orbitals.h"
#include "../abs_pot.h"
#include "../atom.h"
#include "../eigen.h"

#include "../utils.h"
#include "../types.h"

typedef struct {
	ShGrid const* grid;
	Atom const* atom;

	double dt;
	double e_max; // maximum energy

	cdouble* s; // propogation matrix shape = (Nl,Nr,n_evec)
	int n_evec; // number of eigenvec for prop

	ShWavefunc* prop_wf;
} ws_gps_t;

ws_gps_t* ws_gps_alloc(ShGrid const* grid, Atom const* atom, double dt, double e_max);
void ws_gps_free(ws_gps_t* ws);
void ws_gps_calc_s(ws_gps_t* ws, eigen_ws_t const* eigen);
void ws_gps_prop(ws_gps_t const* ws, ShWavefunc* wf);
void ws_gps_prop_common(
		ws_gps_t* ws,
		ShWavefunc* wf,
		UabsCache const* uabs,
		field_t const* field,
		double t
);
