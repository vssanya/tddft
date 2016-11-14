#pragma once

#include "sphere_wavefunc.h"
#include "utils.h"

// az(t) = - Ez(t) - <Ψ|dUdz|Ψ>
// @param dUdz - depends only r. It's dUdz/cos(\theta).
double az(sphere_wavefunc_t const* wf, field_t E, sphere_pot_t dUdz, double t) {
	double dUdz_masked(double r) {
		return dUdz(r)*(1.0 - smoothstep(r, 8.0, 12.0));
	}

	return - E(t) - sphere_wavefunc_cos(wf, dUdz_masked);
}
