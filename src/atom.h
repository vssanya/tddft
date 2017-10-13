#pragma once

#include "grid.h"
#include "sh_wavefunc.h"

#include "types.h"


typedef enum {
  POTENTIAL_SMOOTH,
  POTENTIAL_COULOMB
} potential_type_e;

typedef double (*pot_f)(void*, sh_grid_t const*, int);

typedef struct atom_s {
	int Z; //!< nuclear charge
	int n_orbs; //!< orbitals count
	int* m; //!< z component momentum of orbital
	int* l; //!< full orbital momentum of orbital
	int* n_e; //!< electrons count for each orbital
	pot_f u;
	pot_f dudz;
  potential_type_e u_type;
} atom_t;

int atom_get_count_electrons(atom_t const* atom);
int atom_get_number_ort(atom_t const* atom, int ie);

// Potential
double atom_u_coulomb(atom_t const* atom, sh_grid_t const* grid, int ir) __attribute__((pure));
double atom_dudz_coulomb(atom_t const* atom, sh_grid_t const* grid, int ir) __attribute__((pure));
double atom_u_smooth(atom_t const* atom, sh_grid_t const* grid, int ir) __attribute__((pure));
double atom_dudz_smooth(atom_t const* atom, sh_grid_t const* grid, int ir) __attribute__((pure));
double atom_u_ar_sae(atom_t const* atom, sh_grid_t const* grid, int ir);
double atom_dudz_ar_sae(atom_t const* atom, sh_grid_t const* grid, int ir);

void atom_hydrogen_ground(sh_wavefunc_t* wf);

static atom_t const atom_hydrogen = {
	.Z = 1,
	.n_orbs = 1,
	.m = (int[]){0},
	.l = (int[]){0},
	.n_e = (int[]){1},
	.u    = (pot_f)atom_u_coulomb,
	.dudz = (pot_f)atom_dudz_coulomb,
	.u_type = POTENTIAL_COULOMB,
};

static atom_t const atom_hydrogen_smooth = {
	.Z = 1,
	.n_orbs = 1,
	.m = (int[]){0},
	.l = (int[]){0},
	.n_e = (int[]){1},
	.u    = (pot_f)atom_u_smooth,
	.dudz = (pot_f)atom_dudz_smooth,
	.u_type = POTENTIAL_SMOOTH,
};

// I_p = 0.5791 au
// 1s 2s 2p 3s 3p
// 2  2  6  2  6

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
	.u    = (pot_f)atom_u_coulomb,
	.dudz = (pot_f)atom_dudz_coulomb,
  .u_type = POTENTIAL_COULOMB,
};

static atom_t const atom_argon_sae = {
	.Z = 18,
	.n_orbs = 1,
	.m   = (int[]){0},
	.l   = (int[]){1},
	.n_e = (int[]){1},
	.u    = (pot_f)atom_u_ar_sae,
	.dudz = (pot_f)atom_dudz_ar_sae,
	.u_type = POTENTIAL_COULOMB,
};

static atom_t const atom_argon_ion = {
	.Z = 18,
	.n_orbs = 1,
	.m = (int[]){0},
	.l = (int[]){0},
	.n_e = (int[]){2},
	.u    = (pot_f)atom_u_coulomb,
	.dudz = (pot_f)atom_dudz_coulomb,
	.u_type = POTENTIAL_COULOMB,
};

// I_p = 0.5791 au
// 1s 2s 2p
// 2  2  6
static atom_t const atom_neon = {
	.Z = 10,
	.n_orbs = 4,
	.m    = (int[]){0,0,0,1},
	.l    = (int[]){0,0,1,1},
	.n_e  = (int[]){2,2,2,4},
	.u    = (pot_f)atom_u_coulomb,
	.dudz = (pot_f)atom_dudz_coulomb,
	.u_type = POTENTIAL_COULOMB,
};
