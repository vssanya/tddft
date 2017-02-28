#include "calc.h"
#include "atom.h"
#include "utils.h"


double calc_az(sphere_wavefunc_t const* wf, field_t field, sh_f dudz, double t) {
    return - field_E(field, t) - sphere_wavefunc_cos(wf, dudz);
}

void calc_az_t(int Nt, double a[Nt], sh_workspace_t* ws, sphere_wavefunc_t* wf, field_t field, double dt) {
	double t = 0.0;

	for (int i = 0; i < Nt; ++i) {
        a[i] = calc_az(wf, field, atom_hydrogen_sh_dudz, t);
		sh_workspace_prop(ws, wf, field, t, dt);

		t += dt;
	}
}

double calc_ionization_prob(ks_orbitals_t const* orbs) {
	return 2*orbs->ne - ks_orbitals_norm(orbs);
}
