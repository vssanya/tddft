#include "atom.h"
#include "orbitals.h"

#include <mpi/mpi.h>


int atom_get_count_electrons(atom_t const* atom) {
	int count = 0;
	for (int i=0; i<atom->n_orbs; ++i) {
		count += atom->n_e[i];
	}

	return count;
}

void atom_argon_ort(orbitals_t* orbs) {
	sh_wavefunc_ort_l(0, 3, orbs->wf);

	sh_wavefunc_ort_l(1, 2, &orbs->wf[3]);
	sh_wavefunc_ort_l(1, 2, &orbs->wf[5]);
	//sh_wavefunc_ort_l(1, 2, &orbs->wf[7]);
}

void atom_neon_ort(orbitals_t* orbs) {
	sh_wavefunc_ort_l(0, 2, orbs->wf);
}

inline double sh_u_c(int z, sh_grid_t const* grid, int ir) {
    double const r = sh_grid_r(grid, ir);
	return -z/r;
}

inline double sh_dudz_c(int z, sh_grid_t const* grid, int ir) {
    double const r = sh_grid_r(grid, ir);
	return z/pow(r,2);
}

double atom_argon_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_u_c(18, grid, ir);
}

double atom_argon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_dudz_c(18, grid, ir);
}

double atom_neon_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_u_c(10, grid, ir);
}

double atom_neon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_dudz_c(10, grid, ir);
}

double atom_hydrogen_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_u_c(1, grid, ir);
}

double atom_hydrogen_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_dudz_c(1, grid, ir);
}

void atom_hydrogen_ground(sh_wavefunc_t* wf) {
	assert(wf->m == 0);
	// l = 0
	{
		int const il = 0;
		for (int ir = 0; ir < wf->grid->n[iR]; ++ir) {
			double r = sh_grid_r(wf->grid, ir);
			swf_set(wf, ir, il, 2*r*exp(-r));
		}
	}

	for (int il = 1; il < wf->grid->n[iL]; ++il) {
		for (int ir = 0; ir < wf->grid->n[iR]; ++ir) {
			swf_set(wf, ir, il, 0.0);
		}
	}
}
