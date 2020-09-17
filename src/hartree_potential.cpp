#include "hartree_potential.h"
#include "hartree_potential/hp.h"
#include "sphere_harmonics.h"
#include "utils.h"

#include "grid.h"
#include <math.h>


template<typename Grid>
void _hartree_potential_calc_f_l0(Orbitals<Grid> const* orbs, double* f, Range const& rRange) {
	auto& grid = orbs->grid;
	for (int ie = 0; ie < orbs->atom.countOrbs; ++ie) {
		auto wf = orbs->wf[ie];
		if (wf != nullptr) {
			for (int il = wf->m; il < grid.n[iL]; ++il) {
#pragma omp parallel for
				for (int ir = rRange.start; ir < rRange.end; ++ir) {
					f[ir] += wf->abs_2(ir, il)*orbs->atom.orbs[ie].countElectrons;
				}
			}
		}
	}
}

template<typename Grid>
void _hartree_potential_wf_calc_f_l0(Wavefunc<Grid> const* wf, double* f, Range const& rRange) {
	auto& grid = wf->grid;
	for (int il = wf->m; il < grid.n[iL]; ++il) {
#pragma omp parallel for
		for (int ir = rRange.start; ir < rRange.end; ++ir) {
			f[ir] += wf->abs_2(ir, il);
		}
	}
}

template<typename Grid>
void _hartree_potential_calc_f_l1(Orbitals<Grid> const* orbs, double* f, Range const& rRange) {
	auto& grid = orbs->grid;

	for (int ie = 0; ie < orbs->atom.countOrbs; ++ie) {
		if (orbs->wf[ie] != nullptr) {
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
	for (int ie = 0; ie < orbs->atom.countOrbs; ++ie) {
		if (orbs->wf[ie] != nullptr) {
			auto& wf = *orbs->wf[ie];
			int const n_e = orbs->atom.orbs[ie].countElectrons;
			for (int il = wf.m; il < 2; ++il) {
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
			for (int il = std::max(2, grid.n[iL]-2); il < grid.n[iL]; ++il) {
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
using calc_wf_func_t = void (*)(Wavefunc<Grid> const* wf, double* f, Range const& rRange);

template<typename Grid>
calc_wf_func_t<Grid> const calc_funcs_wf[1] = {
	_hartree_potential_wf_calc_f_l0,
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
void HartreePotential<Grid>::calc(
		Wavefunc<Grid> const* wf,
		int l,
		double* U,
		double* f,
		int order,
		std::optional<Range> rRange
		) {
	assert(l >= 0 && l <= 2);

	auto& grid = wf->grid;

#pragma omp parallel for
	for (int ir = 0; ir < grid.n[iR]; ++ir) {
		f[ir] = 0.0;
	}

	calc_funcs_wf<Grid>[l](wf, f, rRange.value_or(grid.getFullRange(iR)));

	if (order == 3) {
		integrate_rmin_rmax_o3(l, grid, f, U);
	} else {
		integrate_rmin_rmax_o5(l, grid, f, U);
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
	orbs->n_sp(grid[0], n, n_tmp, ylm_cache, 8);

#ifdef _MPI
	if (orbs->mpi_rank == 0 || orbs->mpi_comm == MPI_COMM_NULL)
#endif
	{
		auto func = [orbs, &grid, uxc, n](int ir, int it) -> double {
			double x = mod_grad_n<Grid>(orbs->grid, grid[0], n, ir, it);
			return uxc(n[ir + it*grid->n[iR]], x);
		};

		sh_series<double>(func, l, 0, grid->getGrid2d(), U, ylm_cache, rRange);
	}
}

template<typename Grid>
void XCPotential<Grid>::calc_l0(
		potential_xc_f uxc,
		int l, Orbitals<Grid> const* orbs,
		double* U,
		double* n, // for calc using mpi
		double* n_tmp, // for calc using mpi
		std::optional<Range> rRange
		) {
	auto range = rRange.value_or(orbs->grid.getFullRange(iR));

	if (l==0) {
		orbs->n_l0(n, n_tmp);

#ifdef _MPI
		if (orbs->mpi_rank == 0 || orbs->mpi_comm == MPI_COMM_NULL)
#endif
		{
#pragma omp parallel for
			for (int ir=range.start; ir<range.end; ++ir) {
				double x = mod_dndr(orbs->grid, n, ir);
				U[ir] = uxc(n[ir], x);
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
	double C = 1.0;

	double const amin = 1e-12;
	double const amax = 1e-10;
	if (n > amin && n < amax) {
		C = (n - amin)/(amax - amin);
		C = 1 - (C - 1)*(C - 1);
	} else
	if (n < amin) {
		return 0.0;
	}

	double const betta = 0.05;
	double const ksi = pow(2, 1.0/3.0);

	x *= ksi;
	double res = ux_lda_func(n) + uc_lda_func(n) - betta*x*x*pow(n, 1.0/3.0)/ksi/(1.0 + 3.0*betta*x*log(x + sqrt(x*x + 1.0)));
	if (res != res) {
		return 0.0;
	} else {
		return C*res;
	}
	//return ux_lda_func(n) + uc_lda_func(n) - betta*x*x/(pow(n, 7.0/3.0) + 3.0*betta*x*n*(log(x + sqrt(x*x + pow(n, 8.0/3.0))) - 4.0*log(n)/3.0));
}

double uxc_lda(double n, double x) {
	return ux_lda_func(n) + uc_lda_func(n);
}

double uxc_lda_x(double n, double x) {
	return ux_lda_func(n);
}

template <typename Grid>
double mod_grad_n(Grid const& grid, SpGrid const& sp_grid, double* n, int ir, int it) {
	double dn_dt = 0.0;
	double dn_dr = grid.d_dr(&n[it*grid.n[iR]], ir);

	if (it == 0 || it == sp_grid.n[iT] - 1) {
		dn_dt = 0.0;
	} else {
		dn_dt = (n[ir + (it-1)*grid.n[iR]] - n[ir + (it+1)*grid.n[iR]])/(2*sp_grid.d[iT]);
	}

	double r = grid.r(ir);
	return sqrt(pow(dn_dr,2) + pow(dn_dt/r,2))/pow(n[ir + it*grid.n[iR]], 4.0/3.0);
}

template <typename Grid>
double mod_dndr(Grid const& grid, double* n, int ir) {
	double dn_dr = grid.d_dr(n, ir);
	return abs(dn_dr)/pow(n[ir], 4.0/3.0);
}

template class HartreePotential<ShGrid>;
template class HartreePotential<ShNotEqudistantGrid>;
template class XCPotential<ShGrid>;
template class XCPotential<ShNotEqudistantGrid>;

template <typename Grid>
void calc_V(Wavefunc<Grid>& wf1, int sign_m1, Wavefunc<Grid>& wf2, int sign_m2, int L, cdouble* V) {
	const int m1 = wf1.m*sign_m1;
	const int m2 = wf2.m*sign_m2;
	const int M = m1 - m2;
	for (int ir=0; ir<wf1.grid.n[iR]; ir++) {
		V[ir] = 0.0;
		for (int l1=0; l1<wf1.grid.n[iL]; l1++) {
			//for (int l2=std::max(l1-L, 0); (std::abs(L - l2) <= l1 && l2<wf2.grid.n[iL]); l2++) {
			for (int l2=0; l2<wf2.grid.n[iL]; l2++) {
				V[ir] += sqrt((2*l2 + 1)/(2*l1 + 1))*clebsch_gordan_coef(l2, 0, L, 0, l1, 0)*clebsch_gordan_coef(l2, m2, L, M, l1, m1)*wf1(ir, l1)*conj(wf2(ir, l2));
			}
		}
	}
}

template <typename Grid>
void SlatterPotential<Grid>::calc_l(int l, Orbitals<Grid> const* orbs, double* U) {
	for (int ir=0; ir<orbs->grid.n[iR]; ++ir) {
		U[ir] = 0.0;
	}

	if (l > 0) {
		return;
	}

	double V00[orbs->grid.n[iR]];
	cdouble V[orbs->grid.n[iR]];
	cdouble V_int[orbs->grid.n[iR]];

	for (int ie=0; ie<orbs->atom.countOrbs; ++ie) {
		for (int im = -1; im <= (orbs->wf[ie]->m > 0 ? 1 : 0); im += 2) {
			for (int je=0; je<orbs->atom.countOrbs; ++je) {
				for (int jm = -1; jm <= (orbs->wf[je]->m > 0 ? 1 : 0); jm += 2) {
					for (int L=0; L<2*orbs->grid.n[iL]; ++L) {
						calc_V(orbs->wf[ie][0], im, orbs->wf[je][0], jm, L, V);
						integrate_rmin_rmax_o3(L, orbs->grid, V, V_int);
						for (int ir=0; ir<orbs->grid.n[iR]; ++ir) {
							U[ir] += creal(conj(V[ir])*V_int[ir]);
						}
					}
				}
			}
		}
	}

	orbs->V00(V00, nullptr);
	for (int ir=0; ir<orbs->grid.n[iR]; ++ir) {
		U[ir] = -2*U[ir] / V00[ir];
	}

	//orbs->n_l0(V00, nullptr);
	//for (int ir=0; ir<orbs->grid.n[iR]; ++ir) {
		//U[ir] += uc_lda_func(V00[ir]);/[>sqrt(4*M_PI);//*(1.0 - smoothstep(r, 20, 40));
	//}
}

template <typename Grid>
void SlatterPotential<Grid>::calc_l_gs(int l, Orbitals<Grid> const* orbs, double* U) {
	for (int ir=0; ir<orbs->grid.n[iR]; ++ir) {
		U[ir] = 0.0;
	}

	if (l > 0) {
		return;
	}

	double V00[orbs->grid.n[iR]];
	cdouble V[orbs->grid.n[iR]];
	cdouble V_int[orbs->grid.n[iR]];

	for (int ie=0; ie<orbs->atom.countOrbs; ++ie) {
		for (int im = -1; im <= (orbs->wf[ie]->m > 0 ? 1 : 0); im += 2) {
			for (int je=0; je<orbs->atom.countOrbs; ++je) {
				for (int jm = -1; jm <= (orbs->wf[je]->m > 0 ? 1 : 0); jm += 2) {
					auto wf1 = orbs->wf[ie][0];
					auto wf2 = orbs->wf[je][0];
					int l1 = orbs->atom.orbs[ie].l;
					int l2 = orbs->atom.orbs[je].l;

					for (int ir=0; ir<orbs->grid.n[iR]; ++ir) {
						V[ir] = wf1(ir, l1)*conj(wf2(ir, l2));
					}

					for (int L=0; L<2*orbs->grid.n[iL]; ++L) {
						integrate_rmin_rmax_o3(L, orbs->grid, V, V_int);

						for (int ir=0; ir<orbs->grid.n[iR]; ++ir) {
							U[ir] += creal(pow(clebsch_gordan_coef(l2, 0, L, 0, l1, 0), 2)/(2*l1 + 1)*V_int[ir]*conj(V[ir]));
						}
					}
				}
			}
		}
	}

	orbs->V00(V00, nullptr);
	for (int ir=0; ir<orbs->grid.n[iR]; ++ir) {
		U[ir] = -2*U[ir] / V00[ir];
	}

	//orbs->n_l0(V00, nullptr);
	//for (int ir=0; ir<orbs->grid.n[iR]; ++ir) {
		//U[ir] += uc_lda_func(V00[ir]);/[>sqrt(4*M_PI);//*(1.0 - smoothstep(r, 20, 40));
	//}
}

template class SlatterPotential<ShGrid>;
template class SlatterPotential<ShNotEqudistantGrid>;

template <typename Grid>
SICPotential<Grid>::SICPotential(Grid const& grid):
	grid(grid),
	n_l0(Grid1d(grid.n[iR])),
	n_i_l0(Grid1d(grid.n[iR])),
	uh_l0(Grid1d(grid.n[iR])),
	Utmp(Grid1d(grid.n[iR])),
	tmp(Grid1d(grid.n[iR]))
{}

template <typename Grid>
void SICPotential<Grid>::calc_l(int l, Orbitals<Grid> const* orbs, double* U) {
	double* UcalcData;
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		UcalcData = Utmp.data;
	} else
#endif
	{
		UcalcData = U;
	}

	auto Ucalc = Array1D<double>(UcalcData, Grid1d(grid.n[iR]));
	Ucalc.set(0.0);

	orbs->n_l0(n_l0.data, tmp.data, false);

	for (int ie=0; ie<orbs->atom.countOrbs; ++ie) {
		auto wf = orbs->wf[ie];
		if (wf != nullptr) {
			wf->n_l0(n_i_l0.data);
			HartreePotential<Grid>::calc(wf, 0, uh_l0.data, tmp.data, 3);

#pragma omp parallel for
			for (int ir=0; ir<grid.n[iR]; ++ir) {
				double uxc = ux_lda_func(n_l0(ir)) - ux_lda_func(2*n_i_l0(ir)) - uh_l0(ir);
				Ucalc(ir) += n_i_l0(ir)/(n_l0(ir)/2)*uxc*orbs->atom.orbs[ie].countElectrons/2;
			}
		}
	}

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		MPI_Reduce(Utmp.data, U, grid.n[iR], MPI_DOUBLE, MPI_SUM, 0, orbs->mpi_comm);
	}
#endif
}

template class SICPotential<ShGrid>;
template class SICPotential<ShNotEqudistantGrid>;


template <typename Grid>
CalcPotential<Grid>* CalcPotential<Grid>::get(XCPotentialEnum potentialType, Grid const& grid) {
	switch (potentialType) {
		case LDA_SIC:
			return new SICPotential<Grid>(grid);
		case LDA:
			return new FuncPotential<Grid>(grid, uxc_lda);
		case LDA_X:
			return new FuncPotential<Grid>(grid, uxc_lda_x);
		case LB:
			return new FuncPotential<Grid>(grid, uxc_lb);
	}
}

template class CalcPotential<ShGrid>;
template class CalcPotential<ShNotEqudistantGrid>;
