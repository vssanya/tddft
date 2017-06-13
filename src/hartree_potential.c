#include "hartree_potential.h"
#include "hartree_potential/hp.h"


void _hartree_potential_calc_f_l0(orbitals_t const* orbs, double f[orbs->grid->n[iR]]) {
  sh_grid_t const* grid = orbs->grid;
#ifdef _MPI
  if (orbs->mpi_comm != MPI_COMM_NULL) {
    for (int il = 0; il < grid->n[iL]; ++il) {
      for (int ir = 0; ir < grid->n[iR]; ++ir) {
        f[ir] += swf_get_abs_2(orbs->mpi_wf, ir, il)*orbs->atom->n_e[orbs->mpi_rank];
      }
    }
  } else
#endif
    {
      for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
        sh_wavefunc_t const* wf = orbs->wf[ie];
        for (int il = 0; il < grid->n[iL]; ++il) {
          for (int ir = 0; ir < grid->n[iR]; ++ir) {
            f[ir] += swf_get_abs_2(wf, ir, il)*orbs->atom->n_e[ie];
          }
        }
      }
    }
}

void _hartree_potential_calc_f_l1(orbitals_t const* orbs, double f[orbs->grid->n[iR]]) {
  sh_grid_t const* grid = orbs->grid;
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		sh_wavefunc_t const* wf = orbs->mpi_wf;
		int const n_e = orbs->atom->n_e[orbs->mpi_rank];

		{
			int il = 0;
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += n_e*swf_get(wf, ir, il) *
					clm(il  , wf->m)*conj(swf_get(wf, ir, il+1));
			}
		}

		for (int il = 1; il < grid->n[iL]-1; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += n_e*swf_get(wf, ir, il) * (
						clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1)) +
						clm(il  , wf->m)*conj(swf_get(wf, ir, il+1))
						);
			}
		}

		{
			int il = grid->n[iL]-1;
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += n_e*swf_get(wf, ir, il) *
					clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1));
			}
		}
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			sh_wavefunc_t const* wf = orbs->wf[ie];
			int const n_e = orbs->atom->n_e[ie];

			{
				int il = 0;
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += n_e*swf_get(wf, ir, il) *
						clm(il  , wf->m)*conj(swf_get(wf, ir, il+1));
				}
			}

			for (int il = 1; il < grid->n[iL]-1; ++il) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += n_e*swf_get(wf, ir, il) * (
							clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1)) +
							clm(il  , wf->m)*conj(swf_get(wf, ir, il+1))
							);
				}
			}

			{
				int il = grid->n[iL]-1;
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += n_e*swf_get(wf, ir, il) *
						clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1));
				}
			}
		}
	}
}

void _hartree_potential_calc_f_l2(orbitals_t const* orbs, double f[orbs->grid->n[iR]]) {
  sh_grid_t const* grid = orbs->grid;
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		sh_wavefunc_t const* wf = orbs->mpi_wf;
		int const n_e = orbs->atom->n_e[orbs->mpi_rank];

		for (int il = 0; il < 2; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += n_e*swf_get(wf, ir, il) * (
						plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
						qlm(il, wf->m)*conj(swf_get(wf, ir, il+2))
						);
			}
		}
		for (int il = 2; il < grid->n[iL]-2; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += n_e*swf_get(wf, ir, il) * (
						plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
						qlm(il, wf->m)*conj(swf_get(wf, ir, il+2)) +
						qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
						);
			}
		}
		for (int il = grid->n[iL]-2; il < grid->n[iL]; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += n_e*swf_get(wf, ir, il) * (
						plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
						qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
						);
			}
		}
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			sh_wavefunc_t const* wf = orbs->wf[ie];
			int const n_e = orbs->atom->n_e[ie];
			for (int il = 0; il < 2; ++il) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += n_e*swf_get(wf, ir, il) * (
							plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
							qlm(il, wf->m)*conj(swf_get(wf, ir, il+2))
							);
				}
			}
			for (int il = 2; il < grid->n[iL]-2; ++il) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += n_e*swf_get(wf, ir, il) * (
							plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
							qlm(il, wf->m)*conj(swf_get(wf, ir, il+2)) +
							qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
							);
				}
			}
			for (int il = grid->n[iL]-2; il < grid->n[iL]; ++il) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += n_e*swf_get(wf, ir, il) * (
							plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
							qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
							);
				}
			}
		}
	}
}

typedef void (*calc_func_t)(orbitals_t const* orbs, double* f);
calc_func_t const calc_funcs[3] = {
  _hartree_potential_calc_f_l0,
  _hartree_potential_calc_f_l1,
  _hartree_potential_calc_f_l2
};

void hartree_potential(
		orbitals_t const* orbs,
		int l,
		double U[orbs->grid->n[iR]],
		double U_local[orbs->grid->n[iR]],
		double f[orbs->grid->n[iR]],
		int order
) {
  assert(l >= 0 && l <= 2);

	sh_grid_t const* grid = orbs->grid;

	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		f[ir] = 0.0;
	}

	calc_funcs[l](orbs, f);

	double* U_calc;
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		U_calc = U_local;
	} else
#endif
	{
		U_calc = U;
	}

  if (order == 3) {
    integrate_rmin_rmax_o3(l, grid, f, U_calc);
  } else {
    integrate_rmin_rmax_o5(l, grid, f, U_calc);
  }

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		MPI_Reduce(U_local, U, grid->n[iR], MPI_DOUBLE, MPI_SUM, 0, orbs->mpi_comm);
	}
#endif
}

void hartree_potential_wf_l0(
    sh_wavefunc_t const* wf,
		double U[wf->grid->n[iR]],
		double f[wf->grid->n[iR]],
    int order
) {
	sh_grid_t const* grid = wf->grid;

	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		f[ir] = 0.0;
	}

  for (int il = 0; il < grid->n[iL]; ++il) {
    for (int ir = 0; ir < grid->n[iR]; ++ir) {
      f[ir] += swf_get_abs_2(wf, ir, il);
    }
  }

  if (order == 3) {
    integrate_rmin_rmax_o3(0, grid, f, U);
  } else {
    integrate_rmin_rmax_o5(0, grid, f, U);
  }
}

void uxc_lb(
		int l, orbitals_t const* orbs,
		double U[orbs->grid->n[iR]],
		sp_grid_t const* grid,
		double n[grid->n[iR]*grid->n[iC]], // for calc using mpi
		double n_tmp[grid->n[iR]*grid->n[iC]], // for calc using mpi
		ylm_cache_t const* ylm_cache
		) {
		orbitals_n_sp(orbs, grid, n, n_tmp, ylm_cache);

#ifdef _MPI
		if (orbs->mpi_rank == 0 || orbs->mpi_comm == MPI_COMM_NULL)
#endif
		{
			double func(int ir, int ic) {
				double x = mod_grad_n(grid, n, ir, ic);
				return uxc_lb_func(n[ir + ic*grid->n[iR]], x);
			}

			sh_series(func, l, 0, grid, U, ylm_cache);
		}
}

double uc_lda_func(double n) {
	double const a = (M_LN2 - 1)/(2*M_PI*M_PI);
	double const b = 20.4562557;

	double rs = pow(3/(4*M_PI*n), 1.0/3.0);
	double drs_dn = - rs / (3*n);
	double c1 = 1.0 + b*(1.0/rs + 1.0/(rs*rs));

	return a*log(c1) - n*a/c1*(b*(1.0/pow(rs, 2) + 2.0/pow(rs, 3)))*drs_dn;
}

double ux_lda_func(double n) {
	return - pow(3.0*n/M_PI, 1.0/3.0);
}

double uxc_lb_func(double n, double x) {
	double const betta = 0.05;
	return ux_lda_func(n) + uc_lda_func(n) - betta*x*x*pow(n, 1.0/3.0)/(1.0 + 3.0*betta*x*log(x + sqrt(x*x + 1.0)));
}

double mod_grad_n(sp_grid_t const* grid, double n[grid->n[iR]*grid->n[iC]], int ir, int ic) {
	double dn_dr = 0.0;
	double dn_dc = 0.0;

	if (ir == 0) {
		dn_dr = (n[ir+1 + ic*grid->n[iR]] - n[ir   + ic*grid->n[iR]])/grid->d[iR];
	} else if (ir == grid->n[iR] - 1) {
		dn_dr = (n[ir   + ic*grid->n[iR]] - n[ir-1 + ic*grid->n[iR]])/grid->d[iR];
	} else {
		dn_dr = (n[ir-1 + ic*grid->n[iR]] - n[ir+1 + ic*grid->n[iR]])/(2*grid->d[iR]);
	}

	if (ic == 0) {
		dn_dc = (n[ir + (ic+1)*grid->n[iR]] - n[ir +     ic*grid->n[iR]])/grid->d[iC];
	} else if (ic == grid->n[iC] - 1) {
		dn_dc = (n[ir +     ic*grid->n[iR]] - n[ir + (ic-1)*grid->n[iR]])/grid->d[iC];
	} else {
		dn_dc = (n[ir + (ic-1)*grid->n[iR]] - n[ir + (ic+1)*grid->n[iR]])/(2*grid->d[iC]);
	}

	double c = sp_grid_c(grid, ic);
	double r = sp_grid_r(grid, ir);

	return sqrt(pow(dn_dr,2) + pow(dn_dc/r,2)*(1.0 - c*c))/pow(n[ir + ic*grid->n[iR]], 4.0/3.0);
}
