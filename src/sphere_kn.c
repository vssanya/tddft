#include "sphere_kn.h"

#include <stdlib.h>


sphere_kn_workspace_t* sphere_kn_workspace_alloc(sh_grid_t const* grid, double const dt, sphere_pot_t U, sphere_pot_abs_t Uabs) {
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
void sphere_kn_workspace_prop_ang(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, int l, double E) {
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

	cdouble a_const = 0.25*ws->dt*I*E*clm(l, wf->m);

	double r = 0.0;
	for (int i = 0; i < Nr; ++i) {
		//r += ws->grid->d[iR];
		r = ws->grid->d[iR]*(i+1);

		cdouble const a = a_const*r;

		cdouble const f0 = psi_l0[i] - a*psi_l1[i];
		cdouble const f1 = psi_l1[i] - a*psi_l0[i];
		psi_l0[i] = (f0 - a*f1)/(1.0 - a*a);
		psi_l1[i] = (f1 - a*f0)/(1.0 - a*a);
	}
}

void sphere_kn_workspace_prop_ang_l(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, int lu, int le, double E) {
	/*
	 * Solve Nr equations:
	 * psi(l  , t+dt, r) + a*psi(l+2, t+dt, r) = f0
	 * psi(l+2, t+dt, r) + a*psi(l  , t+dt, r) = f1
	 *
	 * a(r) = a_const*r
	 */

	int const Nr = ws->grid->n[iR];


	cdouble* psi_l0 = &wf->data[le*Nr];
	cdouble* psi_l1 = &wf->data[(le+lu)*Nr];

	cdouble a_const = 0.25*ws->dt*I*E*clm(le, wf->m);

	double r = 0.0;
	for (int i = 0; i < Nr; ++i) {
		//r += ws->grid->d[iR];
		r = ws->grid->d[iR]*(i+1);

		cdouble const a = a_const*r;

		cdouble const f0 = psi_l0[i] - a*psi_l1[i];
		cdouble const f1 = psi_l1[i] - a*psi_l0[i];
		psi_l0[i] = (f0 - a*f1)/(1.0 - a*a);
		psi_l1[i] = (f1 - a*f0)/(1.0 - a*a);
	}
}

// exp(-iΔtHat(l,m, t+Δt/2))
void sphere_kn_workspace_prop_at(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf) {
	double dr = ws->grid->d[iR];
	cdouble dt = ws->dt;
	int Nr = ws->grid->n[iR];

	cdouble c0 = 0.5*dt/(dr*dr);
	cdouble c1 = 0.5*dt;

	for (int l = 0; l < ws->grid->n[iL]; ++l) {
		double r = dr;
		cdouble U = ws->U(r) - I*ws->Uabs(r, ws->grid);

		cdouble al[3];
		cdouble ar[3];
		cdouble f;

		cdouble* psi = &wf->data[l*Nr];

		//al[1] = (1.0 + I*c0*(1.0 - dr/(12 - 10*dr))) + c1*I*(U + l*(l+1)/(2*r*r));
		al[1] = (1.0 + I*c0) + c1*I*(U + l*(l+1)/(2*r*r));
		al[2] = -0.5*I*c0;

		//ar[1] = (1.0 - I*c0*(1.0 - dr/(12 - 10*dr))) - c1*I*(U + l*(l+1)/(2*r*r));
		ar[1] = (1.0 - I*c0) - c1*I*(U + l*(l+1)/(2*r*r));
		ar[2] = 0.5*I*c0;

		f = ar[1]*psi[0] + ar[2]*psi[1];

		ws->alpha[0] = -al[2]/al[1];
		ws->betta[0] = f/al[1];

		al[0] = al[2];
		ar[0] = ar[2];

		for (int i = 1; i < ws->grid->n[iR]; ++i) {
			r += dr;
			U = ws->U(r) - I*ws->Uabs(r, ws->grid);

			al[1] = (1.0 + I*c0) + c1*I*(U + l*(l+1)/(2*r*r));
			ar[1] = (1.0 - I*c0) - c1*I*(U + l*(l+1)/(2*r*r));
			
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
void sphere_kn_workspace_prop_at_v2(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf) {
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
		double r = sh_grid_r(ws->grid, 0);
		cdouble U = ws->U(r) - I*ws->Uabs(r, ws->grid);

		cdouble al[3];
		cdouble ar[3];
		cdouble f;

		cdouble* psi = &wf->data[l*Nr];

		al[0] = M2[0] + 0.5*I*dt*(-0.5*d2[0]);
		al[1] = M2_11 + 0.5*I*dt*(-0.5*d2_11 + U + 0.5*l*(l+1)/(r*r));
		al[2] = M2[2] + 0.5*I*dt*(-0.5*d2[2]);

		ar[0] = M2[0] - 0.5*I*dt*(-0.5*d2[0]);
		ar[1] = M2_11 - 0.5*I*dt*(-0.5*d2_11 + U + 0.5*l*(l+1)/(r*r));
		ar[2] = M2[2] - 0.5*I*dt*(-0.5*d2[2]);

		f = ar[1]*psi[0] + ar[2]*psi[1];

		ws->alpha[0] = -al[2]/al[1];
		ws->betta[0] = f/al[1];

		for (int ir = 1; ir < ws->grid->n[iR]; ++ir) {
			r = sh_grid_r(ws->grid, ir);
			U = ws->U(r) - I*ws->Uabs(r, ws->grid);

			al[1] = M2[1] + 0.5*I*dt*(-0.5*d2[1] + U + 0.5*l*(l+1)/(r*r));
			ar[1] = M2[1] - 0.5*I*dt*(-0.5*d2[1] + U + 0.5*l*(l+1)/(r*r));
			
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

void sphere_kn_workspace_prop(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t field, double t) {
	double Et = field_E(field, t + ws->dt/2);

	for (int il = ws->grid->n[iL] - 2; il >= 0; --il) {
		int const l = sh_grid_l(ws->grid, il);
		sphere_kn_workspace_prop_ang(ws, wf, l, Et);
	}

	sphere_kn_workspace_prop_at_v2(ws, wf);

	for (int il = 0; il < ws->grid->n[iL] - 1; ++il) {
		int const l = sh_grid_l(ws->grid, il);
		sphere_kn_workspace_prop_ang(ws, wf, l, Et);
	}
}

void sphere_kn_workspace_prop_orbs(sphere_kn_workspace_t* ws, ks_orbitals_t* orbs, field_t field, double t) {
	for (int ie = 0; ie < orbs->ne; ++ie) {
		sphere_kn_workspace_prop(ws, orbs->wf[ie], field, t);
	}
}
