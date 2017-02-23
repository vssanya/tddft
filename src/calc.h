#pragma once

#include "fields.h"

#include "sphere_wavefunc.h"
#include "sh_workspace.h"


// az(t) = - Ez(t) - <Ψ|dUdz|Ψ>
// @param dUdz - depends only r. It's dUdz/cos(\theta).
double calc_az(sphere_wavefunc_t const* wf, field_t field, sh_f dudz, double t);

// az(t) low frequency
double calc_az_lf(sphere_wavefunc_t const* wf, field_t field, sh_f dudz, double t);

void calc_az_t(
		int Nt, double a[Nt],
		sh_workspace_t* ws,
		sphere_wavefunc_t* wf,
		field_t field,
		double dt
);
