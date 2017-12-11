#include "sae.h"


using namespace tdse::workspace;

SAE::SAE(sh_grid_t const* grid, uabs_sh_t const* uabs, int num_threads) {
	ws_wf = ws_wf_new(grid, uabs, num_threads);
}

SAE::~SAE() {
	ws_wf_del(ws_wf);
}

void SAE::prop(sh_wavefunc_t* wf, atom_t const* atom, field_t const* field, double t, double dt) {
	double Et = field_E(field, t + dt/2);

	ws_wf_prop_at(ws, wf, dt, Ul[0], atom->Z, atom->u_type);
}