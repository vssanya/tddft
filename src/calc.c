#include "calc.h"
#include "hydrogen.h"
#include "utils.h"


double calc_az(sphere_wavefunc_t const* wf, field_t field, sphere_pot_t dudz, double t) {
    return - field_E(field, t) - sphere_wavefunc_cos(wf, dudz);
}

double calc_az_lf(sphere_wavefunc_t const* wf, field_t field, sphere_pot_t dudz, double t) {
    double dudz_masked(sh_grid_t const* grid, int ir, int l, int m) {
        double const r = sh_grid_r(grid, ir);
        return dudz(grid, ir, l, m)*smoothstep(r, 12.0, 16.0);
	}

	return calc_az(wf, field, dudz_masked, t);
}

void calc_az_t(int Nt, double a[Nt], sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t field) {
	double t = 0.0;

	for (int i = 0; i < Nt; ++i) {
        a[i] = calc_az(wf, field, hydrogen_sh_dudz, t);
		sphere_kn_workspace_prop(ws, wf, field, t);

		t += ws->dt;
	}
}
