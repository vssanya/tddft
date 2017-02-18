#pragma once

#include "grid.h"

#include "sphere_wavefunc.h"
#include "ks_orbitals.h"

// Potential
double hydrogen_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double hydrogen_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));

void hydrogen_ground(sphere_wavefunc_t* wf);


// I_p = 0.5791 au
// 1s 2s 2p 3s 3p
// 2  2  6  2  6
void argon_init(ks_orbitals_t* orbs);
void argon_ort(ks_orbitals_t* orbs);

double argon_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double argon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));


// I_p = 0.5791 au
// 1s 2s 2p
// 2  2  6
void neon_init(ks_orbitals_t* orbs);
void neon_ort(ks_orbitals_t* orbs);

double neon_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double neon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
