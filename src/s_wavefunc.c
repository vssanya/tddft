#include "s_wavefunc.h"

wavefunc_t* wavefunc_new(grid_t const* grid) {
	wavefunc_t* wf = malloc(sizeof(wavefunc_t));
	wf->grid = grid;
	wf->data = calloc(grid_size(sizeof(double)), sizeof(cdouble));

	return wf;
}

void wavefunc_del(wavefunc_t* wf) {
	free(wf->data);
	free(wf->wf);
}

void sp_wavefunc_from_sh(s_wavefunc_t* sp_wf, sphere_wavefunc_t const* sh_wf) {
	for (int ip = 0; ip < sp_wf->grid->n[PHI]) {
		for (int ix = 0; ix < sp_wf->grid->n[X]) {
			for (int il = 0; il < sh_wf->grid->n[L]) {
				for (int ir = 0; ir < sp_wf->grid->n[R]) {
				}
			}
		}
	}
}
