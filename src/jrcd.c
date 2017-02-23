#include "jrcd.h"
#include "calc.h"


double jrcd(
		sh_workspace_t* ws,
		sphere_wavefunc_t* wf,
		field_t E,
		sh_f dUdz,
		int Nt, 
		double dt,
		double t_smooth
) {
	double res = 0.0;

	double t = 0.0;

	double const t_max = Nt*dt;

	for (int i = 0; i < Nt; ++i) {
		res += calc_az(wf, E, dUdz, t)*smoothstep(t_max - t, 0, t_smooth);
		sh_workspace_prop(ws, wf, E, t, dt);
		t += dt;
	}

	return res*dt;
}
