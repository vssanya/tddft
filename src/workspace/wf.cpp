#include <stdlib.h>
#include <stdio.h>

#include "wf.h"
#include "common_alg.h"


void workspace::WfE::prop(ShWavefunc& wf, field_t const* field, double t, double dt) {
	double Et = field_E(field, t + dt/2);

	sh_f Ul[2] = {
            [this](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
                return l*(l+1)/(2*r*r) + atom_cache.u(ir);
			},
            [Et](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
				return r*Et*clm(l,m);
			}
	};


    prop_common(wf, dt, 2, Ul);
	prop_abs(wf, dt);
}

void workspace::WfA::prop(ShWavefunc& wf, field_t const* field, double t, double dt) {
	double At = -field_A(field, t + dt/2);

	sh_f Ul[1] = {
            [this](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
                return l*(l+1)/(2*r*r) + atom_cache.u(ir);
			},
	};

	sh_f Al[2] = {
        [At](ShGrid const* grid, int ir, int l, int m) -> double {
				return At*clm(l,m);
		},
        [At](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
				return At*(l+1)*clm(l,m)/r;
		}
	};


    prop_common(wf, dt, 1, Ul, Al);
	prop_abs(wf, dt);
}

template<>
void workspace::WavefuncWS<ShNotEqudistantGrid>::prop_at_Odr4(Wavefunc<ShNotEqudistantGrid>& wf, cdouble dt, sh_f Ul) {
	int const Nr = grid.n[iR];

	int const Z = atom_cache.atom.Z;


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

#pragma omp for private(U, al, ar, f)
	for (int l = wf.m; l < wf.grid->n[iL]; ++l) {
		int tid = omp_get_thread_num();

		cdouble* alpha_tid = &alpha[tid*Nr];
		cdouble* betta_tid = &betta[tid*Nr];

		cdouble* psi = &wf(0,l);

		cdouble const idt_2 = 0.5*I*dt;

		{
			int ir = 0;

			U[1] = Ul(&grid, ir  , l, wf.m);
			U[2] = Ul(&grid, ir+1, l, wf.m);

			double dr1 = grid.dr(ir);
			double dr2 = grid.dr(ir+1);

			if (l == 0 && atom_cache.atom.potentialType == Atom::POTENTIAL_COULOMB) {
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

			f = ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha_tid[0] = -al[2]/al[1];
			betta_tid[0] = f/al[1];
		}

		for (int ir = 1; ir < Nr-1; ++ir) {
			U[0] = U[1];
			U[1] = U[2];
			U[2] = Ul(&grid, ir+1, l, wf.m);

			double dr1 = grid.dr(ir);
			double dr2 = grid.dr(ir+1);

			for (int i = 0; i < 3; ++i) {
				al[i] = M2[i](dr1, dr2)*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i](dr1, dr2);
				ar[i] = M2[i](dr1, dr2)*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i](dr1, dr2);
			}

			cdouble c = al[1] + al[0]*alpha_tid[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha_tid[ir] = - al[2] / c;
			betta_tid[ir] = (f - al[0]*betta_tid[ir-1]) / c;
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

			cdouble c = al[1] + al[0]*alpha_tid[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir];

			betta_tid[ir] = (f - al[0]*betta_tid[ir-1]) / c;
		}

		psi[Nr-1] = betta_tid[Nr-1];
		for (int ir = Nr-2; ir >= 0; --ir) {
			psi[ir] = alpha_tid[ir]*psi[ir+1] + betta_tid[ir];
		}
	}
}
