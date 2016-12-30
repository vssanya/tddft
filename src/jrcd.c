#include "jrcd.h"


// az(t) = - Ez(t) - <Ψ|dUdz|Ψ>
// @param dUdz - depends only r. It's dUdz/cos(\theta).
double az(sphere_wavefunc_t const* wf, field_t field, sphere_pot_t dUdz, double t) {
	double dUdz_masked(double r) {
		return dUdz(r)*smoothstep(r, 12, 16.0);
	}

	return - field_E(field, t) - sphere_wavefunc_cos(wf, dUdz_masked);
}

/* 
 * jrcd = Ng \int_{0}^{T} az dt 
 * @return jrcd / Ng
 * */
double jrcd(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t E, sphere_pot_t dUdz, int Nt) {
	double res = 0.0;
	double t = 0.0;

	for (int i = 0; i < Nt; ++i) {
		res += az(wf, E, dUdz, t);

		sphere_kn_workspace_prop(ws, wf, E, t);

		t += ws->dt;
	}

	return res*ws->dt;
}
