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
    double const r = sh_grid_r(grid, ir);
	double const r_dr = r + dr;

	F[0] = pow(r/r_dr, l+1)*(F[0] + f[ir]/r);
	F[1] = pow(r_dr/r, l  )*(F[1] - f[ir]/r);

	return (F[0] + F[1])*dr;
}

void hartree_potential_l0(orbitals_t const* orbs, double U[orbs->grid->n[iR]], double U_local[orbs->grid->n[iR]], double f[orbs->grid->n[iR]]) {
	sh_grid_t const* grid = orbs->grid;

	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		f[ir] = 0.0;
	}

	if (orbs->mpi_comm != MPI_COMM_NULL) {
			for (int il = 0; il < grid->n[iL]; ++il) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += swf_get_abs_2(orbs->mpi_wf, ir, il)*orbs->atom->n_e[orbs->mpi_rank];
				}
			}
	} else {
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			sh_wavefunc_t const* wf = orbs->wf[ie];
			for (int il = 0; il < grid->n[iL]; ++il) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += swf_get_abs_2(wf, ir, il)*orbs->atom->n_e[ie];
				}
			}
		}
	}

	double* U_calc;
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		U_calc = U_local;
	} else {
		U_calc = U;
	}

	double F[2];
	U_calc[0] = F_first(F, 0, grid, f);
    for (int ir = 0; ir < grid->n[iR]; ++ir) {
		U_calc[ir] = F_next(F, 0, ir, grid, f);
	}

	if (orbs->mpi_comm != MPI_COMM_NULL) {
		MPI_Allreduce(U_local, U, grid->n[iR], MPI_DOUBLE, MPI_SUM, orbs->mpi_comm);
	}
}

void hartree_potential_wf_l0(sh_wavefunc_t const* wf, double U[wf->grid->n[iR]], double f[wf->grid->n[iR]]) {
	sh_grid_t const* grid = wf->grid;

	for (int il = 0; il < grid->n[iL]; ++il) {
		for (int ir = 0; ir < grid->n[iR]; ++ir) {
			f[ir] += swf_get_abs_2(wf, ir, il);
		}
	}

	double F[2];
	U[0] = 2*F_first(F, 0, grid, f);
    for (int ir = 0; ir < grid->n[iR]; ++ir) {
		U[ir] = 2*F_next(F, 0, ir, grid, f);
	}
}

void hartree_potential_l1(orbitals_t const* orbs, double U[orbs->grid->n[iR]], double f[orbs->grid->n[iR]]) {
	sh_grid_t const* grid = orbs->grid;

	for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
        sh_wavefunc_t const* wf = orbs->wf[ie];

		{
			int il = 0;
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += swf_get(wf, ir, il) *
						clm(il  , wf->m)*conj(swf_get(wf, ir, il+1));
			}
		}

		for (int il = 1; il < grid->n[iL]-1; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += swf_get(wf, ir, il) * (
						clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1)) +
						clm(il  , wf->m)*conj(swf_get(wf, ir, il+1))
                        );
			}
		}

		{
			int il = grid->n[iL]-1;
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += swf_get(wf, ir, il) *
						clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1));
			}
		}
	}

	double F[2];
	U[0] = 2*F_first(F, 0, orbs->grid, f);
	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		U[ir] = 2*F_next(F, 0, ir, orbs->grid, f);
	}
}

void hartree_potential_l2(
		orbitals_t const* orbs,
		double U[orbs->grid->n[iR]],
		double f[orbs->grid->n[iR]]
) {
	sh_grid_t const* grid = orbs->grid;

	for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
        sh_wavefunc_t const* wf = orbs->wf[ie];
        for (int il = 0; il < 2; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
                f[ir] += swf_get(wf, ir, il) * (
						plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
						qlm(il, wf->m)*conj(swf_get(wf, ir, il+2))
				);
			}
		}
        for (int il = 2; il < grid->n[iL]-2; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
                f[ir] += swf_get(wf, ir, il) * (
						plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
						qlm(il, wf->m)*conj(swf_get(wf, ir, il+2)) +
						qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
				);
			}
		}
        for (int il = grid->n[iL]-2; il < grid->n[iL]; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
                f[ir] += swf_get(wf, ir, il) * (
						plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
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

void ux_lda(
		int l, orbitals_t const* orbs,
		double U[orbs->grid->n[iR]],
		sp_grid_t const* grid,
		double n[grid->n[iR]*grid->n[iC]], // for calc using mpi
		double n_tmp[grid->n[iR]*grid->n[iC]], // for calc using mpi
		ylm_cache_t const* ylm_cache
) {
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		orbitals_n_sp(orbs, grid, n, n_tmp, ylm_cache);
		if (orbs->mpi_rank == 0) {
			ux_lda_n(l, grid, n, U, ylm_cache);
		}
		MPI_Bcast(U, orbs->grid->n[iR], MPI_DOUBLE, 0, orbs->mpi_comm);
	} else
#endif
	{
		double func(int ir, int ic) {
			return - pow(3/M_PI*orbitals_n(orbs, grid, (int[2]){ir, ic}, ylm_cache), 1.0/3.0);
		}

		sh_series(func, l, 0, grid, U, ylm_cache);
	}
}

void ux_lda_n(
		int l,
		sp_grid_t const* grid,
		double n[grid->n[iR]*grid->n[iC]],
		double U[grid->n[iR]],
		ylm_cache_t const* ylm_cache
) {
	double func(int ir, int ic) {
		return - pow(3/M_PI*n[ir + ic*grid->n[iR]], 1.0/3.0);
	}

	sh_series(func, l, 0, grid, U, ylm_cache);
}
