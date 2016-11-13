#pragma once

#include "sphere_grid.h"
#include "sphere_wavefunc.h"

typedef double (*sphere_pot_t)(double r);

typedef struct {
	double dt;

	sphere_grid_t const* grid;
	sphere_pot_t U;

	cdouble* b;
	cdouble* f;

	cdouble* alpha;
	cdouble* betta;
} sphere_kn_workspace_t;

sphere_kn_workspace_t* sphere_kn_workspace_alloc(sphere_grid_t const* grid, double const dt, sphere_pot_t U) {
	sphere_kn_workspace_t* ws = malloc(sizeof(sphere_kn_workspace_t));

	ws->grid = grid;
	ws->dt = dt;

	ws->U = U;

	ws->b = malloc(sizeof(cdouble)*grid->Nr);
	ws->f = malloc(sizeof(cdouble)*grid->Nr);

	ws->alpha = malloc(sizeof(cdouble)*grid->Nr);
	ws->betta = malloc(sizeof(cdouble)*grid->Nr);

	return ws;
}

void sphere_kn_workspace_free(sphere_kn_workspace_t* ws) {
	free(ws->b);
	free(ws->f);
	free(ws->alpha);
	free(ws->betta);
	free(ws);
}

void sphere_kn_workspace_prop(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf) {
	double dr = ws->grid->dr;
	cdouble dt = -I*ws->dt;
	int Nr = ws->grid->Nr;

	cdouble c0 = 0.5*dt/(dr*dr);
	cdouble c1 = 0.5*dt;

	for (int l = 0; l < ws->grid->Nl; ++l) {
		double r = dr;
		double U = ws->U(r);

		cdouble al[3];
		cdouble ar[3];
		cdouble f;

		cdouble* psi = &wf->data[l*Nr];

		al[1] = (1.0 + I*c0*(1.0 - dr/(12 - 10*dr))) + c1*I*(U + l*(l+1)/(2*r*r));
		al[2] = -0.5*I*c0;

		ar[1] = (1.0 - I*c0*(1.0 - dr/(12 - 10*dr))) - c1*I*(U + l*(l+1)/(2*r*r));
		ar[2] = 0.5*I*c0;

		f = ar[1]*psi[0] + ar[2]*psi[1];

		ws->alpha[0] = -al[2]/al[1];
		ws->betta[0] = f/al[1];

		al[0] = al[2];
		ar[0] = ar[2];

		for (int i = 1; i < ws->grid->Nr; ++i) {
			r += dr;
			U = ws->U(r);

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
