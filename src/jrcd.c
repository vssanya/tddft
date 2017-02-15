#include "jrcd.h"
#include "calc.h"


/* 
 * jrcd = Ng \int_{0}^{T} az dt 
 * @return jrcd / Ng
 * */
double jrcd(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t E, sphere_pot_t dUdz, int Nt, double t_smooth) {
	double res = 0.0;

	double t = 0.0;

	double const t_max = Nt*ws->dt;

	for (int i = 0; i < Nt; ++i) {
		res += calc_az(wf, E, dUdz, t)*smoothstep(t_max - t, 0, t_smooth);

		sphere_kn_workspace_prop(ws, wf, E, t);

		t += ws->dt;
	}

	return res*ws->dt;
}
