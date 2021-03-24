#pragma once

#include "../wavefunc/sh_2d.h"
#include "../wavefunc/sh_3d.h"
#include "../types.h"
#include "../linalg.h"
#include "../array.h"
#include "../atom.h"


template<typename Grid>
void wf_prop_at_Odr4(
		Array<cdouble, Grid, int> psi,
		cdouble dt,
		std::function<double(int)> Ur,
		bool useBorderCondition,
		int Z,
		cdouble* alpha,
		cdouble* betta
		);

void wf_prop_ang_l(
		Array1D<cdouble> psi_l0,
		Array1D<cdouble> psi_l1,
		cdouble dt,
		int l, int m,
		std::function<double(int, int, int)> Ul,
		linalg::matrix_f dot,
		linalg::matrix_f dot_T,
		cdouble const eigenval[2]
);

template<class Grid>
void wf_prop_ang_l_2(
		Wavefunc<Grid>& wf,
		cdouble dt,
		int l,
		int l1,
		std::function<double(int, int, int)> Ul,
		linalg::matrix_f dot,
		linalg::matrix_f dot_T,
		cdouble const eigenval[2]
) {
	int const Nr = wf.grid.n[iR];

	cdouble* psi_l0 = &wf(0, l);
	cdouble* psi_l1 = &wf(0, l+l1);

#pragma omp for
	for (int i = 0; i < Nr; ++i) {
		double const E = Ul(i, l, wf.m);

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
		std::function<double(int, int, int)> Ul
) {

	wf_prop_ang_l(wf(l), wf(l+l1), dt, l, wf.m, Ul, linalg::matrix_bE::dot, linalg::matrix_bE::dot_T, linalg::matrix_bE::eigenval);
}

template<class Grid>
void wf_prop_ang_E_lm(ShWavefunc3D<Grid>& wf, cdouble dt,
		int l, int l1, int m, int m1, double phi,
		std::function<double(int, int, int)> Ul
) {
	auto dot = [phi](cdouble v[2]) { linalg::matrix_bE_3d::dot(v, phi); };
	auto dot_T = [phi](cdouble v[2]) { linalg::matrix_bE_3d::dot_T(v, phi); };

	wf_prop_ang_l(wf.slice(l, m), wf.slice(l+l1, m+m1), dt, l, m, Ul, dot, dot_T, linalg::matrix_bE::eigenval);
}

template<class Grid>
void wf_prop_ang_A_l(Wavefunc<Grid>& wf, cdouble dt, int l, int l1,
		std::function<double(int, int, int)> Ul
) {
	wf_prop_ang_l_2(wf, dt, l, l1, Ul, linalg::matrix_bA::dot, linalg::matrix_bA::dot_T, linalg::matrix_bA::eigenval);
}
