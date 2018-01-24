#include "hartree_potential.h"
#include "hartree_potential/hp.h"
#include "utils.h"


void _hartree_potential_calc_f_l0(orbitals_t const* orbs, double* f) {
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

void _hartree_potential_calc_f_l1(orbitals_t const* orbs, double* f) {
  sh_grid_t const* grid = orbs->grid;
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		sh_wavefunc_t const& wf = *orbs->mpi_wf;
		int const n_e = orbs->atom->n_e[orbs->mpi_rank];

		{
			int il = 0;
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += creal(n_e*wf(ir, il) *
					clm(il, wf.m)*conj(wf(ir, il+1)));
			}
		}

		for (int il = 1; il < grid->n[iL]-1; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += creal(n_e*wf(ir, il) * (
						clm(il-1, wf.m)*conj(wf(ir, il-1)) +
						clm(il  , wf.m)*conj(wf(ir, il+1))
						));
			}
		}

		{
			int il = grid->n[iL]-1;
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += creal(n_e*wf(ir, il) *
					clm(il-1, wf.m)*conj(wf(ir, il-1)));
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
					f[ir] += creal(n_e*swf_get(wf, ir, il) *
						clm(il, wf->m)*conj(swf_get(wf, ir, il+1)));
				}
			}

			for (int il = 1; il < grid->n[iL]-1; ++il) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += creal(n_e*swf_get(wf, ir, il) * (
							clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1)) +
							clm(il  , wf->m)*conj(swf_get(wf, ir, il+1))
							));
				}
			}

			{
				int il = grid->n[iL]-1;
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += creal(n_e*swf_get(wf, ir, il) *
						clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1)));
				}
			}
		}
	}
}

void _hartree_potential_calc_f_l2(orbitals_t const* orbs, double* f) {
  sh_grid_t const* grid = orbs->grid;
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		sh_wavefunc_t const* wf = orbs->mpi_wf;
		int const n_e = orbs->atom->n_e[orbs->mpi_rank];

		for (int il = 0; il < 2; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += creal(n_e*swf_get(wf, ir, il) * (
						plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
						qlm(il, wf->m)*conj(swf_get(wf, ir, il+2))
						));
			}
		}
		for (int il = 2; il < grid->n[iL]-2; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += creal(n_e*swf_get(wf, ir, il) * (
						plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
						qlm(il, wf->m)*conj(swf_get(wf, ir, il+2)) +
						qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
						));
			}
		}
		for (int il = grid->n[iL]-2; il < grid->n[iL]; ++il) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				f[ir] += creal(n_e*swf_get(wf, ir, il) * (
						plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
						qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
						));
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
					f[ir] += creal(n_e*swf_get(wf, ir, il) * (
							plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
							qlm(il, wf->m)*conj(swf_get(wf, ir, il+2))
							));
				}
			}
			for (int il = 2; il < grid->n[iL]-2; ++il) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += creal(n_e*swf_get(wf, ir, il) * (
							plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
							qlm(il, wf->m)*conj(swf_get(wf, ir, il+2)) +
							qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
							));
				}
			}
			for (int il = grid->n[iL]-2; il < grid->n[iL]; ++il) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					f[ir] += creal(n_e*swf_get(wf, ir, il) * (
							plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
							qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
							));
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
		double* U,
		double* U_local,
		double* f,
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
		double* U,
		double* f,
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

void uxc_calc_l(
		potential_xc_f uxc,
		int l, orbitals_t const* orbs,
		double* U,
		sp_grid_t const* grid,
		double* n, // for calc using mpi
		double* n_tmp, // for calc using mpi
		ylm_cache_t const* ylm_cache
		) {
		orbitals_n_sp(orbs, grid, n, n_tmp, ylm_cache);

#ifdef _MPI
		if (orbs->mpi_rank == 0 || orbs->mpi_comm == MPI_COMM_NULL)
#endif
		{
			auto func = [grid, uxc, n](int ir, int ic) -> double {
				double x = mod_grad_n(grid, n, ir, ic);
				return uxc(n[ir + ic*grid->n[iR]], x);
			};

			sh_series(func, l, 0, grid, U, ylm_cache);
		}
}

void uxc_calc_l0(
		potential_xc_f uxc,
		int l, orbitals_t const* orbs,
		double* U,
		sp_grid_t const* grid,
		double* n, // for calc using mpi
		double* n_tmp, // for calc using mpi
		ylm_cache_t const* ylm_cache
		) {
	if (l==0) {
		orbitals_n_l0(orbs, n, n_tmp);

#ifdef _MPI
		if (orbs->mpi_rank == 0 || orbs->mpi_comm == MPI_COMM_NULL)
#endif
		{
			for (int ir=0; ir<grid->n[iR]; ++ir) {
				double x = mod_dndr(grid, n, ir);
				double r = sh_grid_r(orbs->grid, ir);
				double r_max = sh_grid_r_max(orbs->grid);
				U[ir] = uxc(n[ir], x)*sqrt(4*M_PI);//*(1.0 - smoothstep(r, 20, 40));
			}
		}
	} else {
#ifdef _MPI
		if (orbs->mpi_rank == 0 || orbs->mpi_comm == MPI_COMM_NULL)
#endif
		{
			for (int ir=0; ir<grid->n[iR]; ++ir) {
				U[ir] = 0.0;
			}
		}
	}
}

double uc_lda_func(double n) {
	double const a = (M_LN2 - 1.0)/(2.0*M_PI*M_PI);
	double const b = 20.4562557;

	double m_1_rs = pow(4.0*M_PI*n/3.0, 1.0/3.0);
	double c1 = 1.0 + b*(m_1_rs + pow(m_1_rs, 2));

	return a*log(c1) + a/c1*b*(m_1_rs + 2.0*pow(m_1_rs, 2))/3.0;
}

double ux_lda_func(double n) {
	return - pow(3.0*n/M_PI, 1.0/3.0);
}

double uxc_lb(double n, double x) {
	double const betta = 0.05;
	if (n < 1e-20) {
		return ux_lda_func(n) + uc_lda_func(n);
	}
	return ux_lda_func(n) + uc_lda_func(n) - betta*x*x*pow(n, 1.0/3.0)/(1.0 + 3.0*betta*x*log(x + sqrt(x*x + 1.0)));
	//return ux_lda_func(n) + uc_lda_func(n) - betta*x*x/(pow(n, 7.0/3.0) + 3.0*betta*x*n*(log(x + sqrt(x*x + pow(n, 8.0/3.0))) - 4.0*log(n)/3.0));
}

double uxc_lda(double n, double x) {
	return ux_lda_func(n) + uc_lda_func(n);
}

double uxc_lda_x(double n, double x) {
	return ux_lda_func(n);
}

double mod_grad_n(sp_grid_t const* grid, double* n, int ir, int ic) {
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

double mod_dndr(sp_grid_t const* grid, double* n, int ir) {
	double dn_dr = 0.0;

	if (ir == 0) {
		dn_dr = (n[ir+1] - n[ir])/grid->d[iR];
	} else if (ir == grid->n[iR] - 1) {
		dn_dr = (n[ir] - n[ir-1])/grid->d[iR];
	} else {
		dn_dr = (n[ir-1] - n[ir+1])/(2*grid->d[iR]);
	}

	return sqrt(pow(dn_dr,2))/pow(n[ir], 4.0/3.0);
}