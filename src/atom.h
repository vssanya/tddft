#pragma once

#include "grid.h"
#include "sh_wavefunc.h"

#include "types.h"


typedef enum {
  POTENTIAL_SMOOTH,
  POTENTIAL_COULOMB
} potential_type_e;

typedef struct atom_s {
	int Z; //!< nuclear charge
	int n_orbs; //!< orbitals count
	int* m; //!< z component momentum of orbital
	int* l; //!< full orbital momentum of orbital
	int* n_e; //!< electrons count for each orbital
	sh_f u;
	sh_f dudz;
  potential_type_e u_type;
} atom_t;

int atom_get_count_electrons(atom_t const* atom);
int atom_get_number_ort(atom_t const* atom, int ie);

// Potential
double atom_hydrogen_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double atom_hydrogen_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));

double atom_hydrogen_sh_u_smooth(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double atom_hydrogen_sh_dudz_smooth(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));

void atom_hydrogen_ground(sh_wavefunc_t* wf);

static atom_t const atom_hydrogen = {
	.Z = 1,
	.n_orbs = 1,
	.m = (int[]){0},
	.l = (int[]){0},
	.n_e = (int[]){1},
	.u    = atom_hydrogen_sh_u,
	.dudz = atom_hydrogen_sh_dudz,
  .u_type = POTENTIAL_COULOMB,
};

static atom_t const atom_hydrogen_smooth = {
	.Z = 1,
	.n_orbs = 1,
	.m = (int[]){0},
	.l = (int[]){0},
	.n_e = (int[]){1},
	.u    = atom_hydrogen_sh_u_smooth,
	.dudz = atom_hydrogen_sh_dudz_smooth,
  .u_type = POTENTIAL_SMOOTH,
};

// I_p = 0.5791 au
// 1s 2s 2p 3s 3p
// 2  2  6  2  6
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
	.u    = atom_argon_sh_u,
	.dudz = atom_argon_sh_dudz,
  .u_type = POTENTIAL_COULOMB,
};

static atom_t const atom_argon_gs = {
	.Z = 18,
	.n_orbs = 5,
	.m = (int[]){0,0,0,0,0},
	.l = (int[]){0,0,0,1,1},
	.n_e = (int[]){2,2,2,6,6},
	.u    = atom_argon_sh_u,
	.dudz = atom_argon_sh_dudz,
	.u_type = POTENTIAL_COULOMB,
};

// I_p = 0.5791 au
// 1s 2s 2p
// 2  2  6
double atom_neon_sh_u(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));
double atom_neon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) __attribute__((pure));

static atom_t const atom_neon = {
	.Z = 10,
	.n_orbs = 5,
	.m = (int[]){0,0,-1,0,1},
	.l = (int[]){0,0, 1,1,1},
	.n_e = (int[]){2,2,2,2,2},
	.u    = atom_neon_sh_u,
	.dudz = atom_neon_sh_dudz,
  .u_type = POTENTIAL_COULOMB,
};
