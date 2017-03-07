#pragma once

#include "grid.h"

#include "sh_wavefunc.h"
#include "orbitals.h"

#include "types.h"


typedef void (*atom_init_f)(orbitals_t* orbs);
typedef void (*atom_ort_f)(orbitals_t* orbs);

typedef struct atom_s {
	int ne;
	atom_init_f init;
	atom_ort_f ort;
	sh_f u;
	sh_f dudz;
} atom_t;

void atom_hydrogen_init(orbitals_t* orbs);
// Potential
double atom_hydrogen_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double atom_hydrogen_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
void atom_hydrogen_ground(sh_wavefunc_t* wf);

static atom_t const atom_hydrogen = {
	.ne = 1,
	.init = atom_hydrogen_init,
	.ort  = NULL,
	.u    = atom_hydrogen_sh_u,
	.dudz = atom_hydrogen_sh_dudz
};

// I_p = 0.5791 au
// 1s 2s 2p 3s 3p
// 2  2  6  2  6
void atom_argon_init(orbitals_t* orbs);
void atom_argon_ort(orbitals_t* orbs);

double atom_argon_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double atom_argon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));

static atom_t const atom_argon = {
	.ne = 9,
	.init = atom_argon_init,
	.ort  = atom_argon_ort,
	.u    = atom_argon_sh_u,
	.dudz = atom_argon_sh_dudz
};

// I_p = 0.5791 au
// 1s 2s 2p
// 2  2  6
void atom_neon_init(orbitals_t* orbs);
void atom_neon_ort(orbitals_t* orbs);

double atom_neon_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double atom_neon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));

static atom_t const atom_neon = {
	.ne = 5,
	.init = atom_neon_init,
	.ort  = atom_neon_ort,
	.u    = atom_neon_sh_u,
	.dudz = atom_neon_sh_dudz
};
