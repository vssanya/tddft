#include "hartree_potential.h"

double F_first(double F[2], int l, sh_grid_t const* grid, double f[grid->n[iR]]) {
    double const dr = grid->d[iR];

    F[0] = 0.0;
    F[1] = 0.0;

	for (int ir = 0; ir < grid->n[iR]; ++ir) {
        double const r = sh_grid_r(grid, ir);
		F[1] += f[ir]*pow(dr/r, l)/r;
	}

	return (F[0] + F[1])*dr;
}

/*!\fn
 * \brief \f[F_l(r, f) = \int dr' \frac{r_<^l}{r_>^{l+1}} f(r')\f]
 * \param[in,out] F
 * */
double F_next(double F[2], int l, int ir, sh_grid_t const* grid, double f[grid->n[iR]]) {
    double const dr = grid->d[iR];
    double const r = ir*dr;
	double const r_dr = r + dr;

	F[0] = pow(r/r_dr, l+1)*(F[0] + f[ir]/r);
	F[1] = pow(r_dr/r, l  )*(F[1] - f[ir]/r);

	return (F[0] + F[1])*dr;
}

void hartree_potential_l0(ks_orbitals_t const* orbs, double U[orbs->grid->n[iR]]) {
	sh_grid_t const* grid = orbs->grid;

	double f[grid->n[iR]];
	for (int ie = 0; ie < orbs->ne; ++ie) {
        sphere_wavefunc_t const* wf = orbs->wf[ie];
        for (int il = 0; il < grid->n[iL]; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
                f[ir] += swf_get_abs_2(wf, ir, il);
			}
		}
	}

	double F[2];
	U[0] = 2*F_first(F, 0, orbs->grid, f);
    for (int ir = 0; ir < grid->n[iR]; ++ir) {
		U[ir] = 2*F_next(F, 0, ir, orbs->grid, f);
	}
}

void hartree_potential_l1(ks_orbitals_t const* orbs, double U[orbs->grid->n[iR]]) {
	sh_grid_t const* grid = orbs->grid;

	double f[grid->n[iR]];
	for (int ie = 0; ie < orbs->ne; ++ie) {
        sphere_wavefunc_t const* wf = orbs->wf[ie];
		for (int il = 0; il < grid->n[iL]; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += swf_get(wf, ir, il) * (
						clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1)) +
						clm(il  , wf->m)*conj(swf_get(wf, ir, il+1))
                        );
			}
		}
	}

	double F[2];
	U[0] = 2*F_first(F, 0, orbs->grid, f);
	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		U[ir] = 2*F_next(F, 0, ir, orbs->grid, f);
	}
}

void hartree_potential_l2(ks_orbitals_t const* orbs, double U[orbs->grid->n[iR]]) {
	sh_grid_t const* grid = orbs->grid;

	double f[grid->n[iR]];
	for (int ie = 0; ie < orbs->ne; ++ie) {
        sphere_wavefunc_t const* wf = orbs->wf[ie];
        for (int il = 0; il < grid->n[iL]; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
                f[ir] += swf_get(wf, ir, il) * (
						plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
						qlm(il, wf->m)*conj(swf_get(wf, ir, il+2)) +
						qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
				);
			}
		}
	}

	double F[2];
	U[0] = 2*F_first(F, 0, orbs->grid, f);
    for (int ir = 0; ir < grid->n[iR]; ++ir) {
		U[ir] = 2*F_next(F, 0, ir, orbs->grid, f);
	}
}

void ux_lda(int l, ks_orbitals_t const* orbs, double U[orbs->wf[0]->grid->n[iR]], sp_grid_t const* sp_grid) {
	double func(int ir, int ic) {
		return pow(ks_orbitals_n(orbs, sp_grid, (int[2]){ir, ic}), 1.0/3.0);
	}

	sh_series(func, l, 0, sp_grid, U);
}
