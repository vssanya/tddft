#pragma once

#pragma once
#include "ks_orbitals.h"


// E_3p = 15.76 eV = 0.5791 au
// 1s 2s 2p 3s 3p
// 2  2  6  2  6
void argon_init(ks_orbitals_t* orbs);
void argon_ort(ks_orbitals_t* orbs);

double argon_sh_u(sh_grid_t const* grid, int ir, int il, int m);
double argon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m);
