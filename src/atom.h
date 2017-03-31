#pragma once

#include "grid.h"
#include "sh_wavefunc.h"

#include "types.h"


typedef struct orbitals_s orbitals_t;
typedef void (*atom_ort_f)(orbitals_t* orbs);

typedef struct atom_s {
	int Z; //!< nuclear charge
	int n_orbs; //!< orbitals count
	int* m; //!< z component momentum of orbital
	int* l; //!< full orbital momentum of orbital
	int* n_e; //!< electrons count for each orbital
	atom_ort_f ort;
	sh_f u;
	sh_f dudz;
} atom_t;

// Potential
double atom_hydrogen_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double atom_hydrogen_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
void atom_hydrogen_ground(sh_wavefunc_t* wf);

static atom_t const atom_hydrogen = {
	.Z = 1,
	.n_orbs = 1,
	.m = (int[]){0},
	.l = (int[]){0},
	.n_e = (int[]){1},
	.ort  = NULL,
	.u    = atom_hydrogen_sh_u,
	.dudz = atom_hydrogen_sh_dudz
};

// I_p = 0.5791 au
// 1s 2s 2p 3s 3p
// 2  2  6  2  6
void atom_argon_ort(orbitals_t* orbs);

double atom_argon_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double atom_argon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));

/* Atom argon. We assume ionization doesn't dependent on sign of m and of electron spin. */
static atom_t const atom_argon = {
	.Z = 18,
//	.n_orbs = 9,
//	.m   = (int[]){0,0,0,-1,-1,0,0,1,1},
//	.l   = (int[]){0,0,0, 1, 1,1,1,1,1},
//	.n_e = (int[]){2,2,2, 2, 2,2,2,2,2},
	.n_orbs = 7,
	.m   = (int[]){0,0,0,0,0,1,1},
	.l   = (int[]){0,0,0,1,1,1,1},
	.n_e = (int[]){2,2,2,2,2,4,4},
	.ort  = atom_argon_ort,
	.u    = atom_argon_sh_u,
	.dudz = atom_argon_sh_dudz
};

// I_p = 0.5791 au
// 1s 2s 2p
// 2  2  6
void atom_neon_ort(orbitals_t* orbs);

double atom_neon_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double atom_neon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));

static atom_t const atom_neon = {
	.Z = 10,
	.n_orbs = 5,
	.m = (int[]){0,0,-1,0,1},
	.l = (int[]){0,0, 1,1,1},
	.n_e = (int[]){2,2,2,2,2},
	.ort  = atom_neon_ort,
	.u    = atom_neon_sh_u,
	.dudz = atom_neon_sh_dudz
};
