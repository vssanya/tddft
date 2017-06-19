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

int atom_get_number_ort(atom_t const* atom, int ie) {
  if (ie == atom->n_orbs) {
    return 1;
  }

  for (int i=ie+1; i<atom->n_orbs; ++i) {
    if (atom->l[ie] != atom->l[i] || atom->m[ie] != atom->m[i]) {
      return i - ie;
    }
  }

  return atom->n_orbs - ie;
}

inline double sh_u_coulomb(int z, sh_grid_t const* grid, int ir) {
    double const r = sh_grid_r(grid, ir);
	return -z/r;
}

inline double sh_dudz_coulomb(int z, sh_grid_t const* grid, int ir) {
  double const r = sh_grid_r(grid, ir);
	return z/pow(r,2);
}

inline double sh_u_smooth(int z, double a, double alpha, sh_grid_t const* grid, int ir) {
  double const r = sh_grid_r(grid, ir);
	return -alpha*pow(cosh(r/a), -2) - tanh(r/a)/r;
}

inline double sh_dudz_smooth(int z, double a, double alpha, sh_grid_t const* grid, int ir) {
  double const r = sh_grid_r(grid, ir);
  double const t = tanh(r/a);
  double const s = pow(cosh(r/a), -2);

  return 2.0*alpha*t*s/a + t/pow(r, 2) - s/(a*r);
}

double atom_argon_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_u_coulomb(18, grid, ir);
}

double atom_argon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_dudz_coulomb(18, grid, ir);
}

double atom_neon_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_u_coulomb(10, grid, ir);
}

double atom_neon_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_dudz_coulomb(10, grid, ir);
}

double atom_hydrogen_sh_u(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_u_coulomb(1, grid, ir);
}

double atom_hydrogen_sh_dudz(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_dudz_coulomb(1, grid, ir);
}

double atom_hydrogen_sh_u_smooth(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_u_smooth(1, 0.3, 2.17, grid, ir);
}

double atom_hydrogen_sh_dudz_smooth(sh_grid_t const* grid, int ir, int il, int m) {
	return sh_dudz_smooth(1, 0.3, 2.17, grid, ir);
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
