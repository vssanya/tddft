#include "sh_workspace.h"

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "hartree_potential.h"
#include "sphere_harmonics.h"
#include "abs_pot.h"

sh_workspace_t* sh_workspace_alloc(
		sh_grid_t const* grid,
		sh_f Uabs,
		int num_threads
) {
	sh_workspace_t* ws = malloc(sizeof(sh_workspace_t));

	ws->grid = grid;

	ws->Uabs = Uabs;

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

void sh_workspace_free(sh_workspace_t* ws) {
	free(ws->alpha);
	free(ws->betta);
	free(ws);
}

/* 
 * [1 + 0.5iΔtH(t+Δt/2)] Ψ(r, t+Δt) = [1 - 0.5iΔtH(t+Δt/2)] Ψ(r, t)
 * */

// exp(-0.5iΔtHang(l,m, t+Δt/2))
// @param E = E(t+dt/2)
void sh_workspace_prop_ang_l(sh_workspace_t* ws, sh_wavefunc_t* wf, cdouble dt, int l, int l1, sh_f Ul) {
	/*
	 * Solve Nr equations:
	 * psi(l   , t+dt, r) + a*psi(l+l1, t+dt, r) = f0
	 * psi(l+l1, t+dt, r) + a*psi(l   , t+dt, r) = f1
	 *
	 * a(r) = a_const*r
	 */

	int const Nr = ws->grid->n[iR];

	cdouble* psi_l0 = swf_ptr(wf, 0, l);
	cdouble* psi_l1 = swf_ptr(wf, 0, l+l1);

	cdouble a_const = 0.25*dt*I;

#pragma omp for
	for (int i = 0; i < Nr; ++i) {
		cdouble const a = a_const*Ul(ws->grid, i, l, wf->m);

		cdouble const f0 = psi_l0[i] - a*psi_l1[i];
		cdouble const f1 = psi_l1[i] - a*psi_l0[i];

		psi_l0[i] = (f0 - a*f1)/(1.0 - a*a);
		psi_l1[i] = (f1 - a*f0)/(1.0 - a*a);
	}
}

// exp(-iΔtHat(l,m, t+Δt/2))
void sh_workspace_prop_at(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		double dt,
		sh_f Ul,
		sh_f Uabs
) {
	double dr = ws->grid->d[iR];
	int Nr = ws->grid->n[iR];

	cdouble c0 = 0.5*dt/(dr*dr);
	cdouble c1 = 0.5*dt;

	for (int l = 0; l < ws->grid->n[iL]; ++l) {
		cdouble U = Ul(ws->grid, 0, l, wf->m) - I*Uabs(ws->grid, 0, l, wf->m);

		cdouble al[3];
		cdouble ar[3];
		cdouble f;

		cdouble* psi = &wf->data[l*Nr];

		//al[1] = (1.0 + I*c0*(1.0 - dr/(12 - 10*dr))) + c1*I*(U + l*(l+1)/(2*r*r));
		al[1] = (1.0 + I*c0) + c1*I*U;
		al[2] = -0.5*I*c0;

		//ar[1] = (1.0 - I*c0*(1.0 - dr/(12 - 10*dr))) - c1*I*(U + l*(l+1)/(2*r*r));
		ar[1] = (1.0 - I*c0) - c1*I*U;
		ar[2] = 0.5*I*c0;

		f = ar[1]*psi[0] + ar[2]*psi[1];

		ws->alpha[0] = -al[2]/al[1];
		ws->betta[0] = f/al[1];

		al[0] = al[2];
		ar[0] = ar[2];

		for (int i = 1; i < ws->grid->n[iR]; ++i) {
			U = Ul(ws->grid, i, l, wf->m) - I*Uabs(ws->grid, i, l, wf->m);

			al[1] = (1.0 + I*c0) + c1*I*U;
			ar[1] = (1.0 - I*c0) - c1*I*U;
			
			cdouble c = al[1] + al[0]*ws->alpha[i-1];
			f = ar[0]*psi[i-1] + ar[1]*psi[i] + ar[2]*psi[i+1];

			ws->alpha[i] = - al[2] / c;
			ws->betta[i] = (f - al[0]*ws->betta[i-1]) / c;
		}

		psi[Nr-1] = ws->betta[Nr-1]/(1 - ws->alpha[Nr-1]);
		for (int i = Nr-2; i >= 0; --i) {
			psi[i] = ws->alpha[i]*psi[i+1] + ws->betta[i];
		}
	}
}

// O(dr^4)
void sh_workspace_prop_at_v2(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		cdouble dt,
		sh_f Ul,
		sh_f Uabs,
		int Z // nuclear charge
) {
	double dr = ws->grid->d[iR];

	int Nr = ws->grid->n[iR];

	cdouble U[3];

	double d2[3];
	d2[0] =  1.0/(dr*dr);
	d2[1] = -2.0/(dr*dr);
	d2[2] =  1.0/(dr*dr);

	const double d2_l0_11 = d2[1]*(1.0 - Z*dr/(12.0 - 10.0*Z*dr));

	double M2[3];
	M2[0] = 1.0/12.0;
	M2[1] = 10.0/12.0;
	M2[2] = 1.0/12.0;

	const double M2_l0_11 = 1.0 + d2_l0_11*(dr*dr)/12.0;

#pragma omp for
	for (int l = 0; l < ws->grid->n[iL]; ++l) {
		int tid = omp_get_thread_num();

		cdouble* alpha = &ws->alpha[tid*ws->grid->n[iR]];
		cdouble* betta = &ws->betta[tid*ws->grid->n[iR]];

		cdouble al[3];
		cdouble ar[3];
		cdouble f;

		cdouble* psi = &wf->data[l*Nr];

		cdouble const idt_2 = 0.5*I*dt;

		{
			int ir = 0;

			U[1] = Ul(ws->grid, ir, l, wf->m) - I*Uabs(ws->grid, ir, l, wf->m);
			U[2] = Ul(ws->grid, ir+1, l, wf->m) - I*Uabs(ws->grid, ir+1, l, wf->m);

			for (int i = 1; i < 3; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
			}

			if (l == 0) {
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
            U[2] = Ul(ws->grid, ir+1, l, wf->m) - I*Uabs(ws->grid, ir+1, l, wf->m);

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

			alpha[ir] = - al[2] / c;
			betta[ir] = (f - al[0]*betta[ir-1]) / c;
		}

		psi[Nr-1] = betta[Nr-1]/(1 - alpha[Nr-1]);
		for (int ir = ws->grid->n[iR]-2; ir >= 0; --ir) {
			psi[ir] = alpha[ir]*psi[ir+1] + betta[ir];
		}
	}
}

/*!
 * \f[U(r,t) = \sum_l U_l(r, t)\f]
 * \param[in] Ul = \f[U_l(r, t=t+dt/2)\f]
 * */
void _sh_workspace_prop(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		cdouble dt,
		int l_max,
		sh_f Ul[l_max],
		sh_f Uabs,
		int Z
) {
#pragma omp parallel num_threads(ws->num_threads)
	{
		for (int l1 = 1; l1 < l_max; ++l1) {
			for (int il = 0; il < ws->grid->n[iL] - l1; ++il) {
				sh_workspace_prop_ang_l(ws, wf, dt, il, l1, Ul[l1]);
			}
		}

		sh_workspace_prop_at_v2(ws, wf, dt, Ul[0], Uabs, Z);

		for (int l1 = l_max-1; l1 > 0; --l1) {
			for (int il = ws->grid->n[iL] - 1 - l1; il >= 0; --il) {
				sh_workspace_prop_ang_l(ws, wf, dt, il, l1, Ul[l1]);
			}
		}
	}
}

void sh_workspace_prop(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		atom_t const* atom,
		field_t field,
		double t,
		double dt
) {
	double Et = field_E(field, t + dt/2);

	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(grid, ir, l, m);
	}

	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return r*Et*clm(l,m);
	}

	_sh_workspace_prop(ws,  wf, dt, 2, (sh_f[3]){Ul0, Ul1}, ws->Uabs, atom->Z);
}

void sh_workspace_prop_img(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		atom_t const* atom,
		double dt
) {
	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(grid, ir, l, m);
	}

	_sh_workspace_prop(ws,  wf, -I*dt, 1, (sh_f[3]){Ul0}, uabs_zero, atom->Z);
}

sh_orbs_workspace_t* sh_orbs_workspace_alloc(
		sh_grid_t const* sh_grid,
		sp_grid_t const* sp_grid,
		sh_f Uabs,
		ylm_cache_t const* ylm_cache,
		int num_threads
) {	
	sh_orbs_workspace_t* ws = malloc(sizeof(sh_orbs_workspace_t));

	ws->sh_grid = sh_grid;
	ws->sp_grid = sp_grid;
	ws->ylm_cache = ylm_cache;

	ws->wf_ws = sh_workspace_alloc(sh_grid, Uabs, num_threads);

	ws->Uh = malloc(sizeof(double)*sh_grid->n[iR]*3);
	ws->Uh_local = malloc(sizeof(double)*sh_grid->n[iR]);

	ws->Uxc = malloc(sizeof(double)*sh_grid->n[iR]*3);

	ws->uh_tmp = malloc(sizeof(double)*sh_grid->n[iR]);

	ws->n_sp = malloc(sizeof(double)*ws->sp_grid->n[iR]*ws->sp_grid->n[iC]);
	ws->n_sp_local = malloc(sizeof(double)*ws->sp_grid->n[iR]*ws->sp_grid->n[iC]);

	return ws;
}

void sh_orbs_workspace_free(sh_orbs_workspace_t* ws) {
	free(ws->n_sp_local);
	free(ws->n_sp);
	free(ws->uh_tmp);
	free(ws->Uxc);
	free(ws->Uh);

	sh_workspace_free(ws->wf_ws);

	free(ws);
}

//void sh_orbs_workspace_prop(
//		sh_orbs_workspace_t* ws,
//		orbitals_t* orbs,
//		field_t field,
//		double t,
//		double dt
//) {
//	for (int l=0; l<3; ++l) {
//		ux_lda(l, orbs, &ws->Uxc[l*ws->sh_grid->n[iR]], ws->sp_grid);
//	}
//
//	hartree_potential_l0(orbs, &ws->Uh[0*ws->sh_grid->n[iR]]);
//	hartree_potential_l1(orbs, &ws->Uh[1*ws->sh_grid->n[iR]]);
//	hartree_potential_l2(orbs, &ws->Uh[2*ws->sh_grid->n[iR]]);
//
//	double Et = field_E(field, t + dt/2);
//
//	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
//		double const r = sh_grid_r(grid, ir);
//		return l*(l+1)/(2*r*r) + ws->wf_ws[0]->U(grid, ir, l, m) + ws->Uh[ir] + ws->Uxc[ir] + plm(l,m)*(ws->Uh[ir + 2*grid->n[iR]] + ws->Uxc[ir + 2*grid->n[iR]]);
//	}
//
//	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
//		double const r = sh_grid_r(grid, ir);
//		return clm(l, m)*(r*Et + ws->Uh[ir + grid->n[iR]] + ws->Uxc[ir + grid->n[iR]]);
//	}
//
//	double Ul2(sh_grid_t const* grid, int ir, int l, int m) {
//		return qlm(l, m)*(ws->Uh[ir + 2*grid->n[iR]] + ws->Uxc[ir + 2*grid->n[iR]]);
//	}
//
//	for (int ie = 0; ie < orbs->ne; ++ie) {
//        _sh_workspace_prop(ws->wf_ws[0], orbs->wf[ie], dt, 3, (sh_f[3]){Ul0, Ul1, Ul2}, ws->wf_ws[0]->Uabs);
//	}
//}

void sh_orbs_workspace_prop(
		sh_orbs_workspace_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		field_t field,
		double t,
		double dt
) {
	for (int l=0; l<1; ++l) {
		ux_lda(l, orbs, &ws->Uxc[l*ws->sh_grid->n[iR]], ws->sp_grid, ws->n_sp, ws->n_sp_local, ws->ylm_cache);
	}

	hartree_potential_l0(orbs, &ws->Uh[0*ws->sh_grid->n[iR]], ws->Uh_local, ws->uh_tmp);

	double Et = field_E(field, t + dt/2);

	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(grid, ir, l, m) + ws->Uh[ir] + ws->Uxc[ir]/sqrt(2*M_PI);
	}

	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return clm(l, m)*r*Et;
	}

	double Ul2(sh_grid_t const* grid, int ir, int l, int m) {
		return qlm(l, m)*(ws->Uh[ir + 2*grid->n[iR]] + ws->Uxc[ir + 2*grid->n[iR]]);
	}

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		_sh_workspace_prop(ws->wf_ws, orbs->mpi_wf, dt, 2, (sh_f[3]){Ul0, Ul1, Ul2}, ws->wf_ws->Uabs, atom->Z);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->ne; ++ie) {
			_sh_workspace_prop(ws->wf_ws, orbs->wf[ie], dt, 2, (sh_f[3]){Ul0, Ul1, Ul2}, ws->wf_ws->Uabs, atom->Z);
		}
	}
}

void sh_orbs_workspace_prop_img(
		sh_orbs_workspace_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		double dt
) {
	for (int l=0; l<1; ++l) {
		ux_lda(l, orbs, &ws->Uxc[l*ws->sh_grid->n[iR]], ws->sp_grid, ws->n_sp, ws->n_sp_local, ws->ylm_cache);
	}

	hartree_potential_l0(orbs, &ws->Uh[0*ws->sh_grid->n[iR]], ws->Uh_local, ws->uh_tmp);
//	hartree_potential_l1(orbs, &ws->Uh[1*ws->sh_grid->n[iR]], ws->uh_tmp);
//	hartree_potential_l2(orbs, &ws->Uh[2*ws->sh_grid->n[iR]], ws->uh_tmp);

	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(grid, ir, l, m) + ws->Uh[ir] + ws->Uxc[ir]/sqrt(2*M_PI);// + plm(l,m)*ws->Uh[ir + 2*grid->n[iR]];
	}

	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
		return clm(l, m)*ws->Uh[ir + grid->n[iR]];
	}

	double Ul2(sh_grid_t const* grid, int ir, int l, int m) {
		return qlm(l, m)*ws->Uh[ir + 2*grid->n[iR]];
	}

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		_sh_workspace_prop(ws->wf_ws, orbs->mpi_wf, -I*dt, 1, (sh_f[3]){Ul0, Ul1, Ul2}, uabs_zero, atom->Z);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->ne; ++ie) {
			_sh_workspace_prop(ws->wf_ws, orbs->wf[ie], -I*dt, 1, (sh_f[3]){Ul0, Ul1, Ul2}, uabs_zero, atom->Z);
		}
	}
}
