#pragma once

#include "grid.h"
#include "types.h"
#include "atom.h"


struct  eigen_ws_t {
    ShGrid const* grid;
	double* evec; // eigenvec shape = (Nl,Nr,Ne)
	double* eval; // eigenval shape = (Nl,Ne)
};

eigen_ws_t* eigen_ws_alloc(ShGrid const* grid);
void eigen_ws_free(eigen_ws_t* ws);

void eigen_calc(eigen_ws_t* ws, sh_f u, int Z);
void eigen_calc_for_atom(eigen_ws_t* ws, AtomCache<ShGrid> const* atom);
int eigen_get_n_with_energy(eigen_ws_t const* ws, double energy);

double eigen_eval(eigen_ws_t const* ws, int il, int ie);
double eigen_evec(eigen_ws_t const* ws, int il, int ir, int ie);
