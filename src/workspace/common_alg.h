#pragma once

#include "../sh_wavefunc.h"
#include "../types.h"
#include "../linalg.h"

template<class Grid>
void wf_prop_ang_l(
		Wavefunc<Grid>& wf,
		cdouble dt,
		int l, int l1,
		typename Wavefunc<Grid>::sh_f Ul,
		linalg::matrix_f dot,
		linalg::matrix_f dot_T,
		cdouble const eigenval[2]
) {
	int const Nr = wf.grid->n[iR];

	cdouble* psi_l0 = &wf(0, l);
	cdouble* psi_l1 = &wf(0, l+l1);

#pragma omp for
	for (int i = 0; i < Nr; ++i) {
		double const E = Ul(wf.grid, i, l, wf.m);

		cdouble x[2] = {psi_l0[i], psi_l1[i]};

		dot(x);
		x[0] *= cexp(I*E*dt*eigenval[0]);
		x[1] *= cexp(I*E*dt*eigenval[1]);
		dot_T(x);

		psi_l0[i] = x[0];
		psi_l1[i] = x[1];
	}
}

template<class Grid>
void wf_prop_ang_l_2(
		Wavefunc<Grid>& wf,
		cdouble dt,
		int l,
		int l1,
		typename Wavefunc<Grid>::sh_f Ul,
		linalg::matrix_f dot,
		linalg::matrix_f dot_T,
		cdouble const eigenval[2]
) {
	int const Nr = wf.grid->n[iR];

	cdouble* psi_l0 = &wf(0, l);
	cdouble* psi_l1 = &wf(0, l+l1);

#pragma omp for
	for (int i = 0; i < Nr; ++i) {
		double const E = Ul(wf.grid, i, l, wf.m);

		cdouble x[2] = {psi_l0[i], psi_l1[i]};

		dot(x);
		x[0] *= cexp(E*dt*eigenval[0]);
		x[1] *= cexp(E*dt*eigenval[1]);
		dot_T(x);

		psi_l0[i] = x[0];
		psi_l1[i] = x[1];
	}
}

template<class Grid>
void wf_prop_ang_E_l(Wavefunc<Grid>& wf, cdouble dt, int l, int l1,
		typename Wavefunc<Grid>::sh_f Ul
) {
	wf_prop_ang_l(wf, dt, l, l1, Ul, linalg::matrix_bE::dot, linalg::matrix_bE::dot_T, linalg::matrix_bE::eigenval);
}

template<class Grid>
void wf_prop_ang_A_l(Wavefunc<Grid>& wf, cdouble dt, int l, int l1,
		typename Wavefunc<Grid>::sh_f Ul
) {
	wf_prop_ang_l_2(wf, dt, l, l1, Ul, linalg::matrix_bA::dot, linalg::matrix_bA::dot_T, linalg::matrix_bA::eigenval);
}
