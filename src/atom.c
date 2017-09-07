#include "atom.h"
#include "orbitals.h"


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

double atom_u_coulomb(atom_t const* atom, sh_grid_t const* grid, int ir) {
    double const r = sh_grid_r(grid, ir);
	return -atom->Z/r;
}

double atom_dudz_coulomb(atom_t const* atom, sh_grid_t const* grid, int ir) {
  double const r = sh_grid_r(grid, ir);
	return atom->Z/pow(r,2);
}

double atom_u_smooth(atom_t const* atom, sh_grid_t const* grid, int ir) {
	double const a = 0.3;
	double const alpha = 2.17;

	double const r = sh_grid_r(grid, ir);
	return -alpha*pow(cosh(r/a), -2) - tanh(r/a)/r;
}

double atom_dudz_smooth(atom_t const* atom, sh_grid_t const* grid, int ir) {
	double const a = 0.3;
	double const alpha = 2.17;

  double const r = sh_grid_r(grid, ir);
  double const t = tanh(r/a);
  double const s = pow(cosh(r/a), -2);

  return 2.0*alpha*t*s/a + t/pow(r, 2) - s/(a*r);
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
