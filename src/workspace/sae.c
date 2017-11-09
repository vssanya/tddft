#include <stdlib.h>

#include "sae.h"


ws_sae_t* ws_sae_new(sh_grid_t const* grid, uabs_sh_t const* uabs, int num_threads) {
	ws_sae_t* ws = malloc(sizeof(ws_sae_t));
	ws->ws_wf = ws_wf_new(grid, uabs, num_threads);
}
