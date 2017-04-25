#pragma once

#include "grid.h"
#include "types.h"
#include "atom.h"


typedef struct {
	sh_grid_t const* grid;
	double* evec; // eigenvec shape = (Nl,Nr,Ne)
	double* eval; // eigenval shape = (Nl,Ne)
} eigen_ws_t;

eigen_ws_t* eigen_ws_alloc(sh_grid_t const* grid);
void eigen_ws_free(eigen_ws_t* ws);

void eigen_calc(eigen_ws_t* ws, sh_f u, int Z);
void eigen_calc_for_atom(eigen_ws_t* ws, atom_t const* atom);
int eigen_get_n_with_energy(eigen_ws_t const* ws, double energy);

inline double eigen_eval(eigen_ws_t const* ws, int il, int ie) {
	int const Nr = ws->grid->n[iR];
	return ws->eval[ie + il*Nr];
}

inline double eigen_evec(eigen_ws_t const* ws, int il, int ir, int ie) {
	int const Nr = ws->grid->n[iR];
	return ws->evec[ie + ir*Nr + il*Nr*Nr];
}
