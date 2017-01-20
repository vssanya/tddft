#include "calc.h"


void calc_a(int Nt, double a[Nt], sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t field) {
	double t = 0.0;

	for (int i = 0; i < Nt; ++i) {
        a[i] = - field_E(field, t) - sphere_wavefunc_cos(wf, hydrogen_sh_dUdz);
		sphere_kn_workspace_prop(ws, wf, field, t);

		t += ws->dt;
	}
}
