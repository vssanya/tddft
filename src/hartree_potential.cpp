#include "hartree_potential.h"
#include "hartree_potential/hp.h"
#include "utils.h"

#include "grid.h"


template<typename Grid>
void _hartree_potential_calc_f_l0(Orbitals<Grid> const* orbs, double* f, Range const& rRange) {
	auto& grid = orbs->grid;
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		auto& wf = *orbs->mpi_wf;
		for (int il = wf.m; il < grid.n[iL]; ++il) {
#pragma omp parallel for
			for (int ir = rRange.start; ir < rRange.end; ++ir) {
				f[ir] += wf.abs_2(ir, il)*orbs->atom.orbs[orbs->mpi_rank].countElectrons;
			}
		}
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom.countOrbs; ++ie) {
			auto& wf = *orbs->wf[ie];
			for (int il = wf.m; il < grid.n[iL]; ++il) {
#pragma omp parallel for
				for (int ir = rRange.start; ir < rRange.end; ++ir) {
					f[ir] += wf.abs_2(ir, il)*orbs->atom.orbs[ie].countElectrons;
				}
			}
		}
	}
}

template<typename Grid>
void _hartree_potential_calc_f_l1(Orbitals<Grid> const* orbs, double* f, Range const& rRange) {
	auto& grid = orbs->grid;
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		auto& wf = *orbs->mpi_wf;
		int const n_e = orbs->atom.orbs[orbs->mpi_rank].countElectrons;

		{
			int il = wf.m;
#pragma omp parallel for
			for (int ir = rRange.start; ir < rRange.end; ++ir) {
				f[ir] += creal(n_e*wf(ir, il) *
						clm(il, wf.m)*conj(wf(ir, il+1)));
			}
		}

		for (int il = wf.m + 1; il < grid.n[iL]-1; ++il) {
#pragma omp parallel for
			for (int ir = rRange.start; ir < rRange.end; ++ir) {
				f[ir] += creal(n_e*wf(ir, il) * (
							clm(il-1, wf.m)*conj(wf(ir, il-1)) +
							clm(il  , wf.m)*conj(wf(ir, il+1))
							));
			}
		}

		{
			int il = grid.n[iL]-1;
#pragma omp parallel for
			for (int ir = rRange.start; ir < rRange.end; ++ir) {
				f[ir] += creal(n_e*wf(ir, il) *
						clm(il-1, wf.m)*conj(wf(ir, il-1)));
			}
		}
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom.countOrbs; ++ie) {
			auto& wf = *orbs->wf[ie];
			int const n_e = orbs->atom.orbs[ie].countElectrons;

			{
				int il = wf.m;
#pragma omp parallel for
				for (int ir = rRange.start; ir < rRange.end; ++ir) {
					f[ir] += creal(n_e*wf(ir, il) *
							clm(il, wf.m)*conj(wf(ir, il+1)));
				}
			}

			for (int il = wf.m+1; il < grid.n[iL]-1; ++il) {
#pragma omp parallel for
				for (int ir = rRange.start; ir < rRange.end; ++ir) {
					f[ir] += creal(n_e*wf(ir, il) * (
								clm(il-1, wf.m)*conj(wf(ir, il-1)) +
								clm(il  , wf.m)*conj(wf(ir, il+1))
								));
				}
			}

			{
				int il = grid.n[iL]-1;
#pragma omp parallel for
				for (int ir = rRange.start; ir < rRange.end; ++ir) {
					f[ir] += creal(n_e*wf(ir, il) *
							clm(il-1, wf.m)*conj(wf(ir, il-1)));
				}
			}
		}
	}
}

template<typename Grid>
void _hartree_potential_calc_f_l2(Orbitals<Grid> const* orbs, double* f, Range const& rRange) {
	auto& grid = orbs->grid;
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		const auto wf = *orbs->mpi_wf;
		int const n_e = orbs->atom.orbs[orbs->mpi_rank].countElectrons;

		for (int il = wf.m; il < wf.m + 2; ++il) {
#pragma omp parallel for
			for (int ir = rRange.start; ir < rRange.end; ++ir) {
				f[ir] += creal(n_e*wf(ir, il) * (
							plm(il, wf.m)*conj(wf(ir, il)) +
							qlm(il, wf.m)*conj(wf(ir, il+2))
							));
			}
		}
		for (int il = wf.m + 2; il < grid.n[iL]-2; ++il) {
#pragma omp parallel for
			for (int ir = rRange.start; ir < rRange.end; ++ir) {
				f[ir] += creal(n_e*wf(ir, il) * (
							plm(il,   wf.m)*conj(wf(ir, il)) +
							qlm(il,   wf.m)*conj(wf(ir, il+2)) +
							qlm(il-2, wf.m)*conj(wf(ir, il-2))
							));
			}
		}
		for (int il = grid.n[iL]-2; il < grid.n[iL]; ++il) {
#pragma omp parallel for
			for (int ir = rRange.start; ir < rRange.end; ++ir) {
				f[ir] += creal(n_e*wf(ir, il) * (
							plm(il,   wf.m)*conj(wf(ir, il)) +
							qlm(il-2, wf.m)*conj(wf(ir, il-2))
							));
			}
		}
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom.countOrbs; ++ie) {
			auto& wf = *orbs->wf[ie];
			int const n_e = orbs->atom.orbs[ie].countElectrons;
			for (int il = 0; il < 2; ++il) {
#pragma omp parallel for
				for (int ir = rRange.start; ir < rRange.end; ++ir) {
					f[ir] += creal(n_e*wf(ir, il) * (
								plm(il, wf.m)*conj(wf(ir, il)) +
								qlm(il, wf.m)*conj(wf(ir, il+2))
								));
				}
			}
			for (int il = 2; il < grid.n[iL]-2; ++il) {
#pragma omp parallel for
				for (int ir = rRange.start; ir < rRange.end; ++ir) {
					f[ir] += creal(n_e*wf(ir, il) * (
								plm(il,   wf.m)*conj(wf(ir, il)) +
								qlm(il,   wf.m)*conj(wf(ir, il+2)) +
								qlm(il-2, wf.m)*conj(wf(ir, il-2))
								));
				}
			}
			for (int il = grid.n[iL]-2; il < grid.n[iL]; ++il) {
#pragma omp parallel for
				for (int ir = rRange.start; ir < rRange.end; ++ir) {
					f[ir] += creal(n_e*wf(ir, il) * (
								plm(il,   wf.m)*conj(wf(ir, il)) +
								qlm(il-2, wf.m)*conj(wf(ir, il-2))
								));
				}
			}
		}
	}
}

template<typename Grid>
using calc_func_t = void (*)(Orbitals<Grid> const* orbs, double* f, Range const& rRange);

template<typename Grid>
calc_func_t<Grid> const calc_funcs[3] = {
	_hartree_potential_calc_f_l0,
	_hartree_potential_calc_f_l1,
	_hartree_potential_calc_f_l2
};

template<typename Grid>
void HartreePotential<Grid>::calc_int_func(
		Orbitals<Grid> const* orbs,
		int l,
		double* f,
		std::optional<Range> rRange
		) {
	auto& grid = orbs->grid;

#pragma omp parallel for
	for (int ir = 0; ir < grid.n[iR]; ++ir) {
		f[ir] = 0.0;
	}

	calc_funcs<Grid>[l](orbs, f, rRange.value_or(grid.getFullRange(iR)));
}

template<typename Grid>
void HartreePotential<Grid>::calc(
		Orbitals<Grid> const* orbs,
		int l,
		double* U,
		double* U_local,
		double* f,
		int order,
		std::optional<Range> rRange
		) {
	assert(l >= 0 && l <= 2);

	auto& grid = orbs->grid;

#pragma omp parallel for
	for (int ir = 0; ir < grid.n[iR]; ++ir) {
		f[ir] = 0.0;
	}

	calc_funcs<Grid>[l](orbs, f, rRange.value_or(grid.getFullRange(iR)));

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
		MPI_Reduce(U_local, U, grid.n[iR], MPI_DOUBLE, MPI_SUM, 0, orbs->mpi_comm);
	}
#endif
}

template<typename Grid>
void HartreePotential<Grid>::calc_wf_l0(
		Wavefunc<Grid> const* wf,
		double* U,
		double* f,
		int order,
		std::optional<Range> rRange
		) {
	auto& grid = wf->grid;
	auto range = rRange.value_or(grid.getFullRange(iR));

	for (int ir = 0; ir < grid.n[iR]; ++ir) {
		f[ir] = 0.0;
	}

	for (int il = wf->m; il < grid.n[iL]; ++il) {
		for (int ir = range.start; ir < range.end; ++ir) {
			f[ir] += wf->abs_2(ir, il);
		}
	}

	if (order == 3) {
		integrate_rmin_rmax_o3(0, grid, f, U);
	} else {
		integrate_rmin_rmax_o5(0, grid, f, U);
	}
}

template<typename Grid>
void XCPotential<Grid>::calc_l(
		potential_xc_f uxc,
		int l, Orbitals<Grid> const* orbs,
		double* U,
		SpGrid const* grid,
		double* n, // for calc using mpi
		double* n_tmp, // for calc using mpi
		YlmCache const* ylm_cache,
		std::optional<Range> rRange
		) {
	orbs->n_sp(grid[0], n, n_tmp, ylm_cache);

#ifdef _MPI
	if (orbs->mpi_rank == 0 || orbs->mpi_comm == MPI_COMM_NULL)
#endif
	{
		auto func = [&grid, uxc, n](int ir, int ic) -> double {
			double x = mod_grad_n(grid[0], n, ir, ic);
			return uxc(n[ir + ic*grid->n[iR]], x);
		};

		sh_series(func, l, 0, grid, U, ylm_cache, rRange);
	}
}

template<typename Grid>
void XCPotential<Grid>::calc_l0(
		potential_xc_f uxc,
		int l, Orbitals<Grid> const* orbs,
		double* U,
		SpGrid const* grid,
		double* n, // for calc using mpi
		double* n_tmp, // for calc using mpi
		YlmCache const* ylm_cache,
		std::optional<Range> rRange
		) {
	auto range = rRange.value_or(grid->getFullRange(iR));

	if (l==0) {
		orbs->n_l0(n, n_tmp);

#ifdef _MPI
		if (orbs->mpi_rank == 0 || orbs->mpi_comm == MPI_COMM_NULL)
#endif
		{
#pragma omp parallel for
			for (int ir=range.start; ir<range.end; ++ir) {
				double x = mod_dndr(orbs->grid, n, ir);
				U[ir] = uxc(n[ir], x)*sqrt(4*M_PI);//*(1.0 - smoothstep(r, 20, 40));
			}
		}
	} else {
#ifdef _MPI
		if (orbs->mpi_rank == 0 || orbs->mpi_comm == MPI_COMM_NULL)
#endif
		{
#pragma omp parallel for
			for (int ir=range.start; ir<range.end; ++ir) {
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
	double const ksi = pow(2, 1.0/3.0);

	x *= ksi;
	return ux_lda_func(n) + uc_lda_func(n) - betta*x*x*pow(n, 1.0/3.0)/ksi/(1.0 + 3.0*betta*x*log(x + sqrt(x*x + 1.0)));
	//return ux_lda_func(n) + uc_lda_func(n) - betta*x*x/(pow(n, 7.0/3.0) + 3.0*betta*x*n*(log(x + sqrt(x*x + pow(n, 8.0/3.0))) - 4.0*log(n)/3.0));
}

double uxc_lda(double n, double x) {
	return ux_lda_func(n) + uc_lda_func(n);
}

double uxc_lda_x(double n, double x) {
	return ux_lda_func(n);
}

double mod_grad_n(SpGrid const& grid, double* n, int ir, int ic) {
	double dn_dr = 0.0;
	double dn_dc = 0.0;

	if (ir == 0) {
		dn_dr = (n[ir+1 + ic*grid.n[iR]] - n[ir   + ic*grid.n[iR]])/grid.d[iR];
	} else if (ir == grid.n[iR] - 1) {
		dn_dr = (n[ir   + ic*grid.n[iR]] - n[ir-1 + ic*grid.n[iR]])/grid.d[iR];
	} else {
		dn_dr = (n[ir-1 + ic*grid.n[iR]] - n[ir+1 + ic*grid.n[iR]])/(2*grid.d[iR]);
	}

	if (ic == 0) {
		dn_dc = (n[ir + (ic+1)*grid.n[iR]] - n[ir +     ic*grid.n[iR]])/grid.d[iC];
	} else if (ic == grid.n[iC] - 1) {
		dn_dc = (n[ir +     ic*grid.n[iR]] - n[ir + (ic-1)*grid.n[iR]])/grid.d[iC];
	} else {
		dn_dc = (n[ir + (ic-1)*grid.n[iR]] - n[ir + (ic+1)*grid.n[iR]])/(2*grid.d[iC]);
	}

	double c = grid.c(ic);
	double r = grid.r(ir);

	return sqrt(pow(dn_dr,2) + pow(dn_dc/r,2)*(1.0 - c*c))/pow(n[ir + ic*grid.n[iR]], 4.0/3.0);
}

template <typename Grid>
double mod_dndr(Grid const& grid, double* n, int ir) {
	double dn_dr = grid.d_dr(n, ir);
	return sqrt(pow(dn_dr,2))/pow(n[ir], 4.0/3.0);
}

template class HartreePotential<ShGrid>;
template class HartreePotential<ShNotEqudistantGrid>;
template class XCPotential<ShGrid>;
template class XCPotential<ShNotEqudistantGrid>;
