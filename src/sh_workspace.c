#include "sh_workspace.h"

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "hartree_potential.h"
#include "sphere_harmonics.h"
#include "abs_pot.h"


gps_ws_t* gps_ws_alloc(sh_grid_t const* grid, atom_t const* atom, double dt, double e_max) {
	gps_ws_t* ws = malloc(sizeof(gps_ws_t));

	ws->grid = grid;
	ws->atom = atom;

	ws->dt = dt;
	ws->e_max = e_max;

	ws->s = NULL;

	ws->prop_wf = sh_wavefunc_new(grid, 0);

	return ws;
}

void gps_ws_free(gps_ws_t* ws) {
	if (ws->s != NULL) {
		free(ws->s);
	}
	sh_wavefunc_del(ws->prop_wf);
	free(ws);
}

void gps_ws_calc_s(gps_ws_t* ws, eigen_ws_t const* eigen) {
	int const Nr = ws->grid->n[iR];
	int const Nl = ws->grid->n[iL];

	int const Ne = eigen_get_n_with_energy(eigen, ws->e_max);

	ws->s = calloc(Nl*Nr*Nr, sizeof(cdouble));

#pragma omp parallel for collapse(3)
	for (int il = 0; il < Nl; ++il) {
		for (int ir1 = 0; ir1 < Nr; ++ir1) {
			for (int ir2 = 0; ir2 < Nr; ++ir2) {
				for (int ie = 0; ie < Ne; ++ie) {
					ws->s[ir2 + Nr*(ir1 + il*Nr)] += cexp(-I*ws->dt*eigen_eval(eigen, il, ie))*eigen_evec(eigen, il, ir1, ie)*eigen_evec(eigen, il, ir2, ie);
				}
			}
		}
	}
}

void gps_ws_prop(gps_ws_t const* ws, sh_wavefunc_t* wf) {
	int const Nr = ws->grid->n[iR];
	int const Nl = ws->grid->n[iL];

#pragma omp parallel for
	for (int il = 0; il < Nl; ++il) {
		cdouble* psi = swf_ptr(ws->prop_wf, 0, il);

		for (int ir1 = 0; ir1 < Nr; ++ir1) {
			psi[ir1] = 0.0;
			for (int ir2 = 0; ir2 < Nr; ++ir2) {
				psi[ir1] += ws->s[ir2 + (ir1 + il*Nr)*Nr]*swf_get(wf, ir2, il);
			}
		}
	}

	sh_wavefunc_copy(ws->prop_wf, wf);
}

void gps_ws_prop_common(
		gps_ws_t* ws,
		sh_wavefunc_t* wf,
		uabs_sh_t const* uabs,
		field_t field,
		double t
) {
	double Et = field_E(field, t + ws->dt/2);

	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return r*Et*clm(l,m);
	}

	int l1 = 1;
	for (int il = 0; il < ws->grid->n[iL] - l1; ++il) {
		prop_ang_l(wf, ws->dt*0.5, il, l1, Ul1);
	}

	gps_ws_prop(ws, wf);

	for (int il = ws->grid->n[iL] - 1 - l1; il >= 0; --il) {
		prop_ang_l(wf, ws->dt*0.5, il, l1, Ul1);
	}

	for (int il = 0; il < ws->grid->n[iL]; ++il) {
		for (int ir = 0; ir < ws->grid->n[iR]; ++ir) {
			wf->data[ir + il*ws->grid->n[iR]]*=exp(-uabs_get(uabs, ws->grid, ir, il, wf->m)*ws->dt);
		}
	}
}


sh_workspace_t* sh_workspace_alloc(
		sh_grid_t const* grid,
		uabs_sh_t const* uabs,
		int num_threads
) {
	sh_workspace_t* ws = malloc(sizeof(sh_workspace_t));

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
void prop_ang_l(sh_wavefunc_t* wf, cdouble dt, int l, int l1, sh_f Ul) {
	/*
	 * Solve Nr equations:
	 * psi(l   , t+dt, r) + a*psi(l+l1, t+dt, r) = f0
	 * psi(l+l1, t+dt, r) + a*psi(l   , t+dt, r) = f1
	 *
	 * a(r) = a_const*r
	 */

	int const Nr = wf->grid->n[iR];

	cdouble* psi_l0 = swf_ptr(wf, 0, l);
	cdouble* psi_l1 = swf_ptr(wf, 0, l+l1);

#pragma omp for
	for (int i = 0; i < Nr; ++i) {
		double const E = Ul(wf->grid, i, l, wf->m);

		cdouble x[2] = {psi_l0[i], psi_l1[i]};
		cdouble xl[2] = {x[0] + x[1], -x[0] + x[1]};

		xl[0] *= cexp(-I*E*dt);
		xl[1] *= cexp( I*E*dt);

		psi_l0[i] = (xl[0] - xl[1])*0.5;
		psi_l1[i] = (xl[0] + xl[1])*0.5;
	}
}

void _sh_workspace_prop_ang_l(sh_workspace_t* ws, sh_wavefunc_t* wf, cdouble dt, int l, int l1, sh_f Ul) {
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

// O(dr^4)
void sh_workspace_prop_at(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		cdouble dt,
		sh_f Ul,
		int Z // nuclear charge
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
void _sh_workspace_prop(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		cdouble dt,
		int l_max,
		sh_f Ul[l_max],
		uabs_sh_t const* uabs,
		int Z
) {
#pragma omp parallel num_threads(ws->num_threads)
	{
		for (int l1 = 1; l1 < l_max; ++l1) {
			for (int il = 0; il < ws->grid->n[iL] - l1; ++il) {
				prop_ang_l(wf, dt*0.5, il, l1, Ul[l1]);
			}
		}

		sh_workspace_prop_at(ws, wf, dt, Ul[0], Z);

		for (int l1 = l_max-1; l1 > 0; --l1) {
			for (int il = ws->grid->n[iL] - 1 - l1; il >= 0; --il) {
				prop_ang_l(wf, dt*0.5, il, l1, Ul[l1]);
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

	_sh_workspace_prop(ws,  wf, dt, 2, (sh_f[3]){Ul0, Ul1}, ws->uabs, atom->Z);
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

	_sh_workspace_prop(ws,  wf, -I*dt, 1, (sh_f[3]){Ul0}, &uabs_zero, atom->Z);
}

sh_orbs_workspace_t* sh_orbs_workspace_alloc(
		sh_grid_t const* sh_grid,
		sp_grid_t const* sp_grid,
		uabs_sh_t const* uabs,
		ylm_cache_t const* ylm_cache,
		int num_threads
) {	
	sh_orbs_workspace_t* ws = malloc(sizeof(sh_orbs_workspace_t));

	ws->sh_grid = sh_grid;
	ws->sp_grid = sp_grid;
	ws->ylm_cache = ylm_cache;

	ws->wf_ws = sh_workspace_alloc(sh_grid, uabs, num_threads);

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

void sh_orbs_workspace_prop(
		sh_orbs_workspace_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		field_t field,
		double t,
		double dt
) {
	for (int l=0; l<1; ++l) {
		uxc_lb(l, orbs, &ws->Uxc[l*ws->sh_grid->n[iR]], ws->sp_grid, ws->n_sp, ws->n_sp_local, ws->ylm_cache);
	}

	hartree_potential_l0(orbs, &ws->Uh[0*ws->sh_grid->n[iR]], ws->Uh_local, ws->uh_tmp);
	hartree_potential_l1(orbs, &ws->Uh[1*ws->sh_grid->n[iR]], ws->Uh_local, ws->uh_tmp);
	hartree_potential_l2(orbs, &ws->Uh[2*ws->sh_grid->n[iR]], ws->Uh_local, ws->uh_tmp);

	double Et = field_E(field, t + dt/2);

	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(grid, ir, l, m) + ws->Uxc[ir]/sqrt(4*M_PI) + ws->Uh[ir] + plm(l,m)*(ws->Uh[ir + 2*grid->n[iR]]);// + sqrt(5)*ws->Uxc[ir + 2*grid->n[iR]]/sqrt(4*M_PI));
	}

	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return clm(l, m)*(r*Et + ws->Uh[ir + grid->n[iR]]);// + sqrt(3)*ws->Uxc[ir + grid->n[iR]]/sqrt(4*M_PI));
	}

	double Ul2(sh_grid_t const* grid, int ir, int l, int m) {
		return qlm(l, m)*(ws->Uh[ir + 2*grid->n[iR]]);// + sqrt(5)*ws->Uxc[ir + 2*grid->n[iR]]/sqrt(4*M_PI));
	}

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		_sh_workspace_prop(ws->wf_ws, orbs->mpi_wf, dt, 3, (sh_f[3]){Ul0, Ul1, Ul2}, ws->wf_ws->uabs, atom->Z);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			_sh_workspace_prop(ws->wf_ws, orbs->wf[ie], dt, 3, (sh_f[3]){Ul0, Ul1, Ul2}, ws->wf_ws->uabs, atom->Z);
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
		uxc_lb(l, orbs, &ws->Uxc[l*ws->sh_grid->n[iR]], ws->sp_grid, ws->n_sp, ws->n_sp_local, ws->ylm_cache);
	}

	hartree_potential_l0(orbs, &ws->Uh[0*ws->sh_grid->n[iR]], ws->Uh_local, ws->uh_tmp);

	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(grid, ir, l, m) + ws->Uh[ir]  + ws->Uxc[ir]/sqrt(4*M_PI);
	}

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		_sh_workspace_prop(ws->wf_ws, orbs->mpi_wf, -I*dt, 1, (sh_f[1]){Ul0}, &uabs_zero, atom->Z);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			_sh_workspace_prop(ws->wf_ws, orbs->wf[ie], -I*dt, 1, (sh_f[1]){Ul0}, &uabs_zero, atom->Z);
		}
	}
}
