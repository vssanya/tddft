#include "atom.h"
#include "orbitals.h"

#include <array>


template<int Z, int N, int S, int np, std::array<double, S*np> const& C, std::array<double, S*np> const& B>
double potential_sgb_u(double r) {
	double res = 0.0;

#pragma unroll
	for (int p=0; p<S; p++) {
#pragma unroll
		for (int k=0; k<np; k++) {
			res += C[k + p*np]*pow(r, p)*exp(-B[k + p*np]*r);
		}
	}

	return - (Z - N + 1 + (N-1)*res) / r;
}

template<int Z, int N, int S, int np, std::array<double, S*np> const& C, std::array<double, S*np> const& B>
double potential_sgb_dudz(double r) {
	double res1 = 0.0;
	double res2 = 0.0;

#pragma unroll
	for (int p=0; p<S; p++) {
#pragma unroll
		for (int k=0; k<np; k++) {
			double tmp = C[k + p*np]*exp(-B[k + p*np]*r);
			res1 += tmp*pow(r, p);
			res2 += tmp*(p*pow(r, p-1) - B[k + p*np]*pow(r, p));
		}
	}

	return (Z - N + 1 + (N-1)*res1) / (r*r) - (N-1)*res2/r;
}

extern constexpr std::array<double, 2*5> rb_B {{
	7.83077875,  2.75163799,
	4.30010258,  0.0,
	43.31975597, 0.0,
	2.93818679,  0.0,
	4.97097146,  0.0
}};

extern constexpr std::array<double, 2*5> rb_C {{
	0.81691787,   0.18308213,
	2.53670563,   0.0,
	-19.56508990, 0.0,
	1.06320272,   0.0,
	-0.99934358,  0.0
}};

double atom_u_rb_sae(atom_t const* atom, sh_grid_t const* grid, int ir) {
	return potential_sgb_u<37, 37, 5, 2, rb_C, rb_B>(sh_grid_r(grid, ir));
}

double atom_dudz_rb_sae(atom_t const* atom, sh_grid_t const* grid, int ir) {
	return potential_sgb_dudz<37, 37, 5, 2, rb_C, rb_B>(sh_grid_r(grid, ir));
}


extern constexpr std::array<double, 2*3> na_B {{
	6.46644991, 2.03040457,
	9.07195947, 1.22049052,
	3.66561470, 3.88900584
}};

extern constexpr std::array<double, 2*3> na_C {{
	0.35071677, 0.64928323,
	1.00486813, -0.05093639,
	1.06629058, 0.70089565,
}};

double atom_u_na_sae(atom_t const* atom, sh_grid_t const* grid, int ir) {
	return potential_sgb_u<11, 11, 3, 2, na_C, na_B>(sh_grid_r(grid, ir));
}

double atom_dudz_na_sae(atom_t const* atom, sh_grid_t const* grid, int ir) {
	return potential_sgb_dudz<11, 11, 3, 2, na_C, na_B>(sh_grid_r(grid, ir));
}


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

double atom_u_ar_sae(atom_t const* atom, sh_grid_t const* grid, int ir) {
	static double const A = 5.4;
	static double const B = 1;
	static double const C = 3.682;

	double const r = sh_grid_r(grid, ir);

	return - (1.0 + (A*exp(-B*r) + (17 - A)*exp(-C*r)))/r;
}

double atom_dudz_ar_sae(atom_t const* atom, sh_grid_t const* grid, int ir) {
	static double const A = 5.4;
	static double const B = 1;
	static double const C = 3.682;

	double const r = sh_grid_r(grid, ir);

	return (1.0 + (  A*exp(-B*r) +   (17 - A)*exp(-C*r)))/(r*r) +
		          (B*A*exp(-B*r) + C*(17 - A)*exp(-C*r))/r;
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

double atom_u_ar_smooth(atom_t const* atom, sh_grid_t const* grid, int ir) {
	double const a = 0.3;
	double const alpha = 3.88;

	double const r = sh_grid_r(grid, ir);
	return -alpha*pow(cosh(r/a), -2) - tanh(r/a)/r;
}

double atom_dudz_ar_smooth(atom_t const* atom, sh_grid_t const* grid, int ir) {
	double const a = 0.3;
	double const alpha = 3.88;

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
