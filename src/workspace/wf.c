#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "wf.h"
#include "common_alg.h"


ws_wf_t* ws_wf_new(
		sh_grid_t const* grid,
		uabs_sh_t const* uabs,
		int num_threads
		) {
	ws_wf_t* ws = malloc(sizeof(ws_wf_t));

	ws->grid = grid;

	ws->uabs = uabs;

#ifdef _OPENMP
	int max_threads = omp_get_max_threads();
	if (num_threads < 1 || num_threads > max_threads) {
		ws->num_threads = max_threads;
	} else {
		ws->num_threads = num_threads;
	}
#else
	ws->num_threads = 1;
#endif

	printf("Create workspace with %d threads\n", ws->num_threads);

	ws->alpha = malloc(sizeof(cdouble)*grid->n[iR]*ws->num_threads);
	ws->betta = malloc(sizeof(cdouble)*grid->n[iR]*ws->num_threads);

	return ws;
}

void ws_wf_del(ws_wf_t* ws) {
	free(ws->alpha);
	free(ws->betta);
	free(ws);
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
void ws_wf_prop_at(
		ws_wf_t* ws,
		sh_wavefunc_t* wf,
		cdouble dt,
		sh_f Ul,
		int Z, // nuclear charge
		potential_type_e u_type
		) {
	double const dr = ws->grid->d[iR];
	double const dr2 = dr*dr;

	int const Nr = ws->grid->n[iR];

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
	for (int l = 0; l < ws->grid->n[iL]; ++l) {
		int tid = omp_get_thread_num();

		cdouble* alpha = &ws->alpha[tid*ws->grid->n[iR]];
		cdouble* betta = &ws->betta[tid*ws->grid->n[iR]];

		cdouble* psi = &wf->data[l*Nr];

		cdouble const idt_2 = 0.5*I*dt;

		{
			int ir = 0;

			U[1] = Ul(ws->grid, ir, l, wf->m);
			U[2] = Ul(ws->grid, ir+1, l, wf->m);

			for (int i = 1; i < 3; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
			}

			if (l == 0 && u_type == POTENTIAL_COULOMB) {
				al[1] = M2_l0_11*(1.0 + idt_2*U[1]) - 0.5*idt_2*d2_l0_11;
				ar[1] = M2_l0_11*(1.0 - idt_2*U[1]) + 0.5*idt_2*d2_l0_11;
			}

			f = ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha[0] = -al[2]/al[1];
			betta[0] = f/al[1];
		}

		for (int ir = 1; ir < ws->grid->n[iR] - 1; ++ir) {
			U[0] = U[1];
			U[1] = U[2];
			U[2] = Ul(ws->grid, ir+1, l, wf->m);

			for (int i = 0; i < 3; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
			}

			cdouble c = al[1] + al[0]*alpha[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha[ir] = - al[2] / c;
			betta[ir] = (f - al[0]*betta[ir-1]) / c;
		}

		{
			int ir = ws->grid->n[iR] - 1;

			U[0] = U[1];
			U[1] = U[2];

			for (int i = 0; i < 2; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
			}

			cdouble c = al[1] + al[0]*alpha[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir];

			betta[ir] = (f - al[0]*betta[ir-1]) / c;
		}

		psi[Nr-1] = betta[Nr-1];
		for (int ir = ws->grid->n[iR]-2; ir >= 0; --ir) {
			psi[ir] = alpha[ir]*psi[ir+1] + betta[ir];
		}
	}
}

/*!
 * \f[U(r,t) = \sum_l U_l(r, t)\f]
 * \param[in] Ul = \f[U_l(r, t=t+dt/2)\f]
 * */
void ws_wf_prop_common(
		ws_wf_t* ws,
		sh_wavefunc_t* wf,
		cdouble dt,
		int l_max,
		sh_f Ul[l_max],
		uabs_sh_t const* uabs,
		int Z,
		potential_type_e u_type
		) {
#pragma omp parallel num_threads(ws->num_threads)
	{
		for (int l1 = 1; l1 < l_max; ++l1) {
			for (int il = 0; il < ws->grid->n[iL] - l1; ++il) {
				wf_prop_ang_l(wf, 0.5*dt, il, l1, Ul[l1]);
			}
		}

		ws_wf_prop_at(ws, wf, dt, Ul[0], Z, u_type);

		for (int l1 = l_max-1; l1 > 0; --l1) {
			for (int il = ws->grid->n[iL] - 1 - l1; il >= 0; --il) {
				wf_prop_ang_l(wf, 0.5*dt, il, l1, Ul[l1]);
			}
		}

#pragma omp for collapse(2)
		for (int il = 0; il < ws->grid->n[iL]; ++il) {
			for (int ir = 0; ir < ws->grid->n[iR]; ++ir) {
				wf->data[ir + il*ws->grid->n[iR]]*=exp(-uabs_get(uabs, ws->grid, ir, il, wf->m)*dt);
			}
		}
	}
}

void ws_wf_prop(
		ws_wf_t* ws,
		sh_wavefunc_t* wf,
		atom_t const* atom,
		field_t const* field,
		double t,
		double dt
		) {
	double Et = field_E(field, t + dt/2);

	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(atom, grid, ir);
	}

	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return r*Et*clm(l,m);
	}

	ws_wf_prop_common(ws,  wf, dt, 2, (sh_f[3]){Ul0, Ul1}, ws->uabs, atom->Z, atom->u_type);
}

void ws_wf_prop_img(
		ws_wf_t* ws,
		sh_wavefunc_t* wf,
		atom_t const* atom,
		double dt
		) {
	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(atom, grid, ir);
	}

	ws_wf_prop_common(ws,  wf, -I*dt, 1, (sh_f[3]){Ul0}, &uabs_zero, atom->Z, atom->u_type);
}
