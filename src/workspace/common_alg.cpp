#include "common_alg.h"
#include <omp.h>

void wf_prop_ang_l(
		Array1D<cdouble> psi_l0,
		Array1D<cdouble> psi_l1,
		cdouble dt,
		int l, int m,
		std::function<double(int, int, int)> Ul,
		linalg::matrix_f dot,
		linalg::matrix_f dot_T,
		cdouble const eigenval[2]
) {
	int const Nr = psi_l0.grid.n;

#pragma omp for
	for (int i = 0; i < Nr; ++i) {
		double const E = Ul(i, l, m);

		cdouble x[2] = {psi_l0(i), psi_l1(i)};

		dot(x);
		x[0] *= cexp(I*E*dt*eigenval[0]);
		x[1] *= cexp(I*E*dt*eigenval[1]);
		dot_T(x);

		psi_l0(i) = x[0];
		psi_l1(i) = x[1];
	}
}

template<typename Grid>
void wf_prop_at_Odr4(
		Array<cdouble, Grid, int> psi,
		cdouble dt,
		std::function<double(int)> Ur,
		bool useBorderCondition,
		int Z,
		cdouble* alpha,
		cdouble* betta
		) {
	auto grid = psi.grid;

	int const Nr = grid.n;

	double dr1 = grid.dr(0);
	double dr2 = grid.dr(1);

	double dr1_dr2 = dr1*(3*dr1 + 4*dr2)*Z - 6*(dr1 + dr2);
	double d2_l0_11 = 2*(dr1+dr2)*(6*dr1 - (3*dr1 - dr2)*(dr1+dr2)*Z)/(dr1*dr1*dr2*dr1_dr2);
	double d2_l_11  = 2*(3*dr1-dr2)*(dr1+dr2)*(dr1+dr2)/(dr1*dr1*dr1*dr2*(3*dr1 + 4*dr2));

	double M2_l0_11 = (dr1+dr2)*(dr1*(dr1+dr2)*(dr1+3*dr2)*Z - 3*(dr1*dr1 + 3*dr2*dr1 + dr2*dr2))/(3*dr1*dr2*dr1_dr2);
	double M2_l0_12 = (-dr1*dr1*dr1*Z + dr1*dr1*(dr2*Z + 3) + dr1*dr2*(2*dr2*Z - 3) - 3*dr2*dr2)/(3*dr2*dr1_dr2);

	double M2_l_11 = (dr1+dr2)*(dr1+dr2)*(dr1+3*dr2)/(3*dr1*dr2*(3*dr1+4*dr2));
	double M2_l_12 = -(dr1-2*dr2)*(dr1+dr2)/(3*dr2*(3*dr1+4*dr2));

	double U[3];
	cdouble al[3];
	cdouble ar[3];
	cdouble f;

	std::function<double(double, double)> M2[3] = {
		[](double d1, double d2) -> double {
			return (d1*d1 + d1*d2 - d2*d2)/(6*d1*(d1+d2));
		},
		[](double d1, double d2) -> double {
			return (d1*d1 + 3*d1*d2 + d2*d2)/(6*d1*d2);
		},
		[](double d1, double d2) -> double {
			return (-d1*d1 + d1*d2 + d2*d2)/(6*d2*(d1+d2));
		}
	};

	std::function<double(double, double)> d2[3] = {
		[](double d1, double d2) -> double {
			return 2.0/(d1*(d1+d2));
		},
		[](double d1, double d2) -> double {
			return -2.0/(d1*d2);
		},
		[](double d1, double d2) -> double {
			return 2.0/(d2*(d1+d2));
		}
	};


	cdouble const idt_2 = 0.5*I*dt;

	{
		int ir = 0;

		U[1] = Ur(ir);
		U[2] = Ur(ir+1);

		double dr1 = grid.dr(ir);
		double dr2 = grid.dr(ir+1);

		if (useBorderCondition) {
			int i = 1;
			al[i] = M2_l0_11*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2_l0_11;
			ar[i] = M2_l0_11*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2_l0_11;

			i = 2;
			al[i] = M2_l0_12*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i](dr1, dr2);
			ar[i] = M2_l0_12*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i](dr1, dr2);
		} else {
			int i = 1;
			al[i] = M2_l_11*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2_l_11;
			ar[i] = M2_l_11*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2_l_11;

			i = 2;
			al[i] = M2_l_12*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i](dr1, dr2);
			ar[i] = M2_l_12*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i](dr1, dr2);
		}

		f = ar[1]*psi(ir) + ar[2]*psi(ir+1);

		alpha[0] = -al[2]/al[1];
		betta[0] = f/al[1];
	}

	for (int ir = 1; ir < Nr-1; ++ir) {
		U[0] = U[1];
		U[1] = U[2];
		U[2] = Ur(ir+1);

		double dr1 = grid.dr(ir);
		double dr2 = grid.dr(ir+1);

		for (int i = 0; i < 3; ++i) {
			al[i] = M2[i](dr1, dr2)*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i](dr1, dr2);
			ar[i] = M2[i](dr1, dr2)*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i](dr1, dr2);
		}

		cdouble c = al[1] + al[0]*alpha[ir-1];
		f = ar[0]*psi(ir-1) + ar[1]*psi(ir) + ar[2]*psi(ir+1);

		alpha[ir] = - al[2] / c;
		betta[ir] = (f - al[0]*betta[ir-1]) / c;
	}

	{
		int ir = Nr-1;

		U[0] = U[1];
		U[1] = U[2];

		double dr1 = grid.dr(ir-1);
		double dr2 = grid.dr(ir);

		for (int i = 0; i < 2; ++i) {
			al[i] = M2[i](dr1, dr2)*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i](dr1, dr2);
			ar[i] = M2[i](dr1, dr2)*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i](dr1, dr2);
		}

		cdouble c = al[1] + al[0]*alpha[ir-1];
		f = ar[0]*psi(ir-1) + ar[1]*psi(ir);

		betta[ir] = (f - al[0]*betta[ir-1]) / c;
	}

	psi(Nr-1) = betta[Nr-1];
	for (int ir = Nr-2; ir >= 0; --ir) {
		psi(ir) = alpha[ir]*psi(ir+1) + betta[ir];
	}
}

template
void wf_prop_at_Odr4<GridNotEq1d>(
		Array<cdouble, GridNotEq1d, int> psi,
		cdouble dt,
		std::function<double(int)> Ur,
		bool useBorderCondition,
		int Z,
		cdouble* alpha,
		cdouble* betta
		);

template
void wf_prop_at_Odr4<Grid1d>(
		Array<cdouble, Grid1d, int> psi,
		cdouble dt,
		std::function<double(int)> Ur,
		bool useBorderCondition,
		int Z,
		cdouble* alpha,
		cdouble* betta
		);
