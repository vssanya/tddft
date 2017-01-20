#include "sphere_kn.h"

#include <stdlib.h>

#include "hartree_potential.h"


sphere_kn_workspace_t* sphere_kn_workspace_alloc(sh_grid_t const* grid, double const dt, sphere_pot_t U, sphere_pot_t Uabs) {
	sphere_kn_workspace_t* ws = malloc(sizeof(sphere_kn_workspace_t));

	ws->grid = grid;
	ws->dt = dt;

	ws->U = U;
	ws->Uabs = Uabs;

	ws->b = malloc(sizeof(cdouble)*grid->n[iR]);
	ws->f = malloc(sizeof(cdouble)*grid->n[iR]);

	ws->alpha = malloc(sizeof(cdouble)*grid->n[iR]);
	ws->betta = malloc(sizeof(cdouble)*grid->n[iR]);

	return ws;
}

void sphere_kn_workspace_free(sphere_kn_workspace_t* ws) {
	free(ws->b);
	free(ws->f);
	free(ws->alpha);
	free(ws->betta);
	free(ws);
}

/* 
 * [1 + 0.5iΔtH(t+Δt/2)] Ψ(r, t+Δt) = [1 - 0.5iΔtH(t+Δt/2)] Ψ(r, t)
 * */

// exp(-0.5iΔtHang(l,m, t+Δt/2))
// @param E = E(t+dt/2)
void sphere_kn_workspace_prop_ang_l1(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, int l, double* U) {
	/*
	 * Solve Nr equations:
	 * psi(l  , t+dt, r) + a*psi(l+1, t+dt, r) = f0
	 * psi(l+1, t+dt, r) + a*psi(l  , t+dt, r) = f1
	 *
	 * a(r) = a_const*r
	 */

	int const Nr = ws->grid->n[iR];


	cdouble* psi_l0 = &wf->data[l*Nr];
	cdouble* psi_l1 = &wf->data[(l+1)*Nr];

	cdouble a_const = 0.25*ws->dt*I*clm(l, wf->m);

	for (int i = 0; i < Nr; ++i) {
		cdouble const a = a_const*U[i];

		cdouble const f0 = psi_l0[i] - a*psi_l1[i];
		cdouble const f1 = psi_l1[i] - a*psi_l0[i];
		psi_l0[i] = (f0 - a*f1)/(1.0 - a*a);
		psi_l1[i] = (f1 - a*f0)/(1.0 - a*a);
	}
}

void sphere_kn_workspace_prop_ang_l(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, int l, int l1, sphere_pot_t Ul) {
	/*
	 * Solve Nr equations:
	 * psi(l   , t+dt, r) + a*psi(l+l1, t+dt, r) = f0
	 * psi(l+l1, t+dt, r) + a*psi(l   , t+dt, r) = f1
	 *
	 * a(r) = a_const*r
	 */

	int const Nr = ws->grid->n[iR];


	cdouble* psi_l0 = &wf->data[l*Nr];
	cdouble* psi_l1 = &wf->data[(l+l1)*Nr];

	cdouble a_const = 0.25*ws->dt*I;

	for (int i = 0; i < Nr; ++i) {
		cdouble const a = a_const*Ul(ws->grid, i, l, wf->m);

		cdouble const f0 = psi_l0[i] - a*psi_l1[i];
		cdouble const f1 = psi_l1[i] - a*psi_l0[i];
		psi_l0[i] = (f0 - a*f1)/(1.0 - a*a);
		psi_l1[i] = (f1 - a*f0)/(1.0 - a*a);
	}
}

// exp(-iΔtHat(l,m, t+Δt/2))
void sphere_kn_workspace_prop_at(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, sphere_pot_t Ul, sphere_pot_t Uabs) {
	double dr = ws->grid->d[iR];
	cdouble dt = ws->dt;
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
void sphere_kn_workspace_prop_at_v2(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, sphere_pot_t Ul, sphere_pot_t Uabs) {
	double dr = ws->grid->d[iR];
	cdouble dt = ws->dt;
	int Nr = ws->grid->n[iR];

	double d2[3];
	d2[0] =  1.0/(dr*dr);
	d2[1] = -2.0/(dr*dr);
	d2[2] =  1.0/(dr*dr);

	const double d2_11 = d2[1]*(1.0 - dr/(12.0 - 10.0*dr));

	double M2[3];
	M2[0] = 1.0/12.0;
	M2[1] = 10.0/12.0;
	M2[2] = 1.0/12.0;

	const double M2_11 = 1.0 + d2_11*(dr*dr)/12.0;
	for (int l = 0; l < ws->grid->n[iL]; ++l) {
        cdouble U = Ul(ws->grid, 0, l, wf->m) - I*Uabs(ws->grid, 0, l, wf->m);

		cdouble al[3];
		cdouble ar[3];
		cdouble f;

		cdouble* psi = &wf->data[l*Nr];

		al[0] = M2[0] + 0.5*I*dt*(-0.5*d2[0]);
		al[1] = M2_11 + 0.5*I*dt*(-0.5*d2_11 + U);
		al[2] = M2[2] + 0.5*I*dt*(-0.5*d2[2]);

		ar[0] = M2[0] - 0.5*I*dt*(-0.5*d2[0]);
		ar[1] = M2_11 - 0.5*I*dt*(-0.5*d2_11 + U);
		ar[2] = M2[2] - 0.5*I*dt*(-0.5*d2[2]);

		f = ar[1]*psi[0] + ar[2]*psi[1];

		ws->alpha[0] = -al[2]/al[1];
		ws->betta[0] = f/al[1];

		for (int ir = 1; ir < ws->grid->n[iR]; ++ir) {
            U = Ul(ws->grid, ir, l, wf->m) - I*Uabs(ws->grid, ir, l, wf->m);

			al[1] = M2[1] + 0.5*I*dt*(-0.5*d2[1] + U);
			ar[1] = M2[1] - 0.5*I*dt*(-0.5*d2[1] + U);
			
			cdouble c = al[1] + al[0]*ws->alpha[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir] + ar[2]*psi[ir+1];

			ws->alpha[ir] = - al[2] / c;
			ws->betta[ir] = (f - al[0]*ws->betta[ir-1]) / c;
		}

		psi[Nr-1] = ws->betta[Nr-1]/(1 - ws->alpha[Nr-1]);
		for (int ir = ws->grid->n[iR]-2; ir >= 0; --ir) {
			psi[ir] = ws->alpha[ir]*psi[ir+1] + ws->betta[ir];
		}
	}
}

/*!
 * \f[U(r,t) = \sum_l U_l(r, t)\f]
 * \param[in] Ul = \f[U_l(r, t=t+dt/2)\f]
 * */
void _sphere_kn_workspace_prop(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, sphere_pot_t Ul[3], sphere_pot_t Uabs) {
    for (int l1 = 1; l1 < 3; ++l1) {
        for (int il = ws->grid->n[iL] - 1 - l1; il >= 0; --il) {
            int const l = sh_grid_l(ws->grid, il);
            sphere_kn_workspace_prop_ang_l(ws, wf, l, l1, Ul[l1]);
        }
    }

	sphere_kn_workspace_prop_at_v2(ws, wf, Ul[0], Uabs);

    for (int l1 = 2; l1 > 0; --l1) {
        for (int il = 0; il < ws->grid->n[iL] - l1; ++il) {
            int const l = sh_grid_l(ws->grid, il);
            sphere_kn_workspace_prop_ang_l(ws, wf, l, l1, Ul[l1]);
        }
    }
}

void sphere_kn_workspace_prop(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t field, double t) {
	double Et = field_E(field, t + ws->dt/2);

	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + ws->U(grid, ir, l, m);
	}

	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return r*Et*clm(l,m);
	}

	double Ul2(sh_grid_t const* grid, int ir, int l, int m) {
		return 0.0;
	}

	_sphere_kn_workspace_prop(ws,  wf, (sphere_pot_t[3]){Ul0, Ul1, Ul2}, ws->Uabs);
}

sphere_kn_orbs_workspace_t* sphere_kn_orbs_workspace_alloc(sh_grid_t const* grid, double const dt, sphere_pot_t U, sphere_pot_t Uabs) {
	sphere_kn_orbs_workspace_t* ws = malloc(sizeof(sphere_kn_workspace_t));
	ws->wf_ws = sphere_kn_workspace_alloc(grid, dt, U, Uabs);
	ws->Uh  = malloc(sizeof(double)*grid->n[iR]*3);
	ws->Uxc = malloc(sizeof(double)*grid->n[iR]*3);
	ws->sp_grid = sp_grid_new((int[3]){grid->n[iR], 16, 1}, sh_grid_r_max(grid));

	return ws;
}

void sphere_kn_orbs_workspace_free(sphere_kn_orbs_workspace_t* ws) {
	sphere_kn_workspace_free(ws->wf_ws);
	sp_grid_del(ws->sp_grid);
	free(ws->Uh);
	free(ws->Uxc);
	free(ws);
}

void sphere_kn_orbs_workspace_prop(sphere_kn_orbs_workspace_t* ws, ks_orbitals_t* orbs, field_t field, double t) {
	for (int l=0; l<3; ++l) {
		ux_lda(l, orbs, &ws->Uxc[l*ws->wf_ws->grid->n[iR]], ws->sp_grid);
	}

	double Et = field_E(field, t + ws->wf_ws->dt/2);

	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + ws->wf_ws->U(grid, ir, l, m) + ws->Uh[ir] + ws->Uxc[ir] + plm(l,m)*(ws->Uh[ir + 2*grid->n[iR]] + ws->Uxc[ir + 2*grid->n[iR]]);
	}

	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return clm(l, m)*(r*Et + ws->Uh[ir + grid->n[iR]] + ws->Uxc[ir + grid->n[iR]]);
	}

	double Ul2(sh_grid_t const* grid, int ir, int l, int m) {
		return qlm(l, m)*(ws->Uh[ir + 2*grid->n[iR]] + ws->Uxc[ir + 2*grid->n[iR]]);
	}

	for (int ie = 0; ie < orbs->ne; ++ie) {
        _sphere_kn_workspace_prop(ws->wf_ws, orbs->wf[ie], (sphere_pot_t[3]){Ul0, Ul1, Ul2}, ws->wf_ws->Uabs);
	}
}
