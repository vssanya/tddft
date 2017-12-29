#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "wf.h"
#include "common_alg.h"
#include "../linalg.h"


workspace::wf_base::wf_base(sh_grid_t const* grid, uabs_sh_t const* uabs, int num_threads):
	grid(grid),
	uabs(uabs),
	num_threads(num_threads)
{
#ifdef _OPENMP
	int max_threads = omp_get_max_threads();
	if (num_threads < 1 || num_threads > max_threads) {
		num_threads = max_threads;
	}
#else
	num_threads = 1;
#endif

	alpha = new cdouble[grid->n[iR]*num_threads];
	betta = new cdouble[grid->n[iR]*num_threads];
}

workspace::wf_base::~wf_base() {
	delete[] alpha;
	delete[] betta;
}

void workspace::wf_base::prop_mix(sh_wavefunc_t& wf, sh_f Al, double dt, int l) {
	int    const Nr = grid->n[iR];
	double const dr = grid->d[iR];

	cdouble* v[2] = {&wf(0,l), &wf(0,l+1)};
	linalg::matrix_dot_vec(Nr, v, linalg::matrix_bE::dot);

	double const glm = dt*Al(grid, 0, l, wf.m)/(4.0*dr);
	const double x = sqrt(3.0) - 2.0;

	linalg::tdm_t M = {(4.0+x)/6.0, {1.0/6.0, 2.0/3.0, 1.0/6.0}, Nr};

	linalg::eq_solve(v[0], M, { x*glm, {-glm, 0.0,  glm}, Nr}, alpha, betta);
	linalg::eq_solve(v[1], M, {-x*glm, { glm, 0.0, -glm}, Nr}, alpha, betta);

	linalg::matrix_dot_vec(Nr, v, linalg::matrix_bE::dot_T);
}

// O(dr^4)
/*
 * \brief Расчет функции \f[\psi(t+dt) = exp(-iH_{at}dt)\psi(t)\f]
 *
 * \f[H_{at} = -0.5\frac{d^2}{dr^2} + U(r, l)\f]
 * \f[exp(iAdt) = \frac{1 - iA}{1 + iA} + O(dt^3)\f]
 *
 * \param[in,out] wf
 *
 */
void workspace::wf_base::prop_at(sh_wavefunc_t& wf, cdouble dt, sh_f Ul, int Z, potential_type_e u_type) {
	int const Nr = grid->n[iR];
	double const dr = grid->d[iR];
	double const dr2 = dr*dr;


	double const d2[3] = {1.0/dr2, -2.0/dr2, 1.0/dr2};
	double const d2_l0_11 = d2[1]*(1.0 - Z*dr/(12.0 - 10.0*Z*dr));

	double const M2[3] = {
		1.0/12.0,
		10.0/12.0,
		1.0/12.0
	};

	const double M2_l0_11 = (1.0 + d2_l0_11*dr2/12.0);

	double U[3];
	cdouble al[3];
	cdouble ar[3];
	cdouble f;

#pragma omp for private(U, al, ar, f)
	for (int l = 0; l < grid->n[iL]; ++l) {
		int tid = omp_get_thread_num();

		cdouble* alpha_tid = &alpha[tid*Nr];
		cdouble* betta_tid = &betta[tid*Nr];

		cdouble* psi = &wf(0,l);

		cdouble const idt_2 = 0.5*I*dt;

		{
			int ir = 0;

			U[1] = Ul(grid, ir  , l, wf.m);
			U[2] = Ul(grid, ir+1, l, wf.m);

			for (int i = 1; i < 3; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
			}

			if (l == 0 && u_type == POTENTIAL_COULOMB) {
				al[1] = M2_l0_11*(1.0 + idt_2*U[1]) - 0.5*idt_2*d2_l0_11;
				ar[1] = M2_l0_11*(1.0 - idt_2*U[1]) + 0.5*idt_2*d2_l0_11;
			}

			f = ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha_tid[0] = -al[2]/al[1];
			betta_tid[0] = f/al[1];
		}

		for (int ir = 1; ir < Nr-1; ++ir) {
			U[0] = U[1];
			U[1] = U[2];
			U[2] = Ul(grid, ir+1, l, wf.m);

			for (int i = 0; i < 3; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
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

			for (int i = 0; i < 2; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
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

void workspace::wf_base::prop_common(sh_wavefunc_t& wf, cdouble dt, int l_max, sh_f* Ul, int Z, potential_type_e u_type, sh_f* Al) {
	int const Nl = grid->n[iL];
#pragma omp parallel
	{
		for (int l1 = 1; l1 < l_max; ++l1) {
			for (int il = 0; il < Nl - l1; ++il) {
				wf_prop_ang_E_l(wf, 0.5*dt, il, l1, Ul[l1]);
			}
		}

		if (Al != nullptr) {
			for (int il=0; il<Nl-1; ++il) {
				wf_prop_ang_A_l(wf, dt*0.5, il, 1, Al[0]);
			}

			for (int il=0; il<Nl-1; ++il) {
				prop_mix(wf, Al[0], creal(dt*0.5), il);
			}
		}

		prop_at(wf, dt, Ul[0], Z, u_type);

		if (Al != nullptr) {
			for (int il=Nl-2; il>=0; --il) {
				prop_mix(wf, Al[0], creal(dt*0.5), il);
			}

			for (int il=Nl-2; il>=0; --il) {
				wf_prop_ang_A_l(wf, dt*0.5, il, 1, Al[0]);
			}
		}

		for (int l1 = l_max-1; l1 > 0; --l1) {
			for (int il = Nl - 1 - l1; il >= 0; --il) {
				wf_prop_ang_E_l(wf, 0.5*dt, il, l1, Ul[l1]);
			}
		}

	}
}

void workspace::wf_base::prop_abs(sh_wavefunc_t& wf, double dt) {
#pragma omp parallel for collapse(2)
	for (int il = 0; il < grid->n[iL]; ++il) {
		for (int ir = 0; ir < grid->n[iR]; ++ir) {
			wf(ir, il) *= exp(-uabs_get(uabs, grid, ir, il, wf.m)*dt);
		}
	}
}

void workspace::wf_base::prop(sh_wavefunc_t& wf, atom_t const* atom, field_t const* field, double t, double dt) {
	double Et = field_E(field, t + dt/2);

	sh_f Ul[2] = {
			[atom](sh_grid_t const* grid, int ir, int l, int m) -> double {
				double const r = sh_grid_r(grid, ir);
				return l*(l+1)/(2*r*r) + atom->u(atom, grid, ir);
			},
			[Et](sh_grid_t const* grid, int ir, int l, int m) -> double {
				double const r = sh_grid_r(grid, ir);
				return r*Et*clm(l,m);
			}
	};


	prop_common(wf, dt, 2, Ul, atom->Z, atom->u_type);

	prop_abs(wf, dt);
}

void workspace::wf_A::prop(sh_wavefunc_t& wf, atom_t const* atom, field_t const* field, double t, double dt) {
	double At = field_A(field, t + dt/2);

	sh_f Ul[1] = {
			[atom](sh_grid_t const* grid, int ir, int l, int m) -> double {
				double const r = sh_grid_r(grid, ir);
				return l*(l+1)/(2*r*r) + atom->u(atom, grid, ir);
			},
	};

	sh_f Al[1] = {
		[At](sh_grid_t const* grid, int ir, int l, int m) -> double {
				return At*clm(l,m);
		}
	};


	prop_common(wf, dt, 1, Ul, atom->Z, atom->u_type, Al);
	prop_abs(wf, dt);
}

void workspace::wf_base::prop_img(sh_wavefunc_t& wf, atom_t const* atom, double dt) {
	sh_f Ul[1] = {
		[atom](sh_grid_t const* grid, int ir, int l, int m) -> double {
			double const r = sh_grid_r(grid, ir);
			return l*(l+1)/(2*r*r) + atom->u(atom, grid, ir);
		}
	};

	prop_common(wf, -I*dt, 1, Ul, atom->Z, atom->u_type);
}
