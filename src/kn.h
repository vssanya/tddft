#pragma once

#include <stdlib.h>

#include "types.h"

typedef double (*pot1d_t)(double x, double t);

typedef struct {
	double dx, dt;
	int N;

	pot1d_t U;

	cdouble* b;
	cdouble* f;

	cdouble* alpha;
	cdouble* betta;
} kn1d_workspace_t;

kn1d_workspace_t* kn1d_workspace_alloc(int const N, double const dx, double const dt, pot1d_t U) {
	kn1d_workspace_t* ws = malloc(sizeof(kn1d_workspace_t));

	ws->dx = dx;
	ws->dt = dt;
	ws->N = N;

	ws->U = U;

	ws->b = malloc(sizeof(cdouble)*N);
	ws->f = malloc(sizeof(cdouble)*N);

	ws->alpha = malloc(sizeof(cdouble)*N);
	ws->betta = malloc(sizeof(cdouble)*N);

	return ws;
}

void kn1d_workspace_free(kn1d_workspace_t* ws) {
	free(ws->b);
	free(ws->f);
	free(ws->alpha);
	free(ws->betta);
	free(ws);
}

void kn1d_workspace_prop(kn1d_workspace_t* ws, cdouble* psi, double t) {
	int    const N  = ws->N;
	double const dt = ws->dt;
	double const dx = ws->dx;

	pot1d_t U = ws->U;

	double x = -(N/2)*dx;
	cdouble c0 = 1.0 + 0.5*I*dt/(dx*dx);
	cdouble c1 = 0.5*I*dt;
	for (int i = 0; i < N; ++i) {
		ws->b[i] = c0 + c1*U(x, t);
		x += dx;
	}

	x = -(N/2)*dx;
	c0 = 0.25*I*dt/(dx*dx);
	c1 = 1-0.5*I*dt/(dx*dx);
	cdouble c2 = -0.5*I*dt;

	{
		int i = 0;
		ws->f[i] = c0*(psi[i+1] + psi[i]  ) + (c1 + c2*U(x, t))*psi[i];
		x += dx;
	}
	for (int i = 1; i < N-1; ++i) {
		ws->f[i] = c0*(psi[i+1] + psi[i-1]) + (c1 + c2*U(x, t))*psi[i];
		x += dx;
	}
	{
		int i = N-1;
		ws->f[i] = c0*(psi[i]   + psi[i-1]) + (c1 + c2*U(x, t))*psi[i];
	}

	cdouble a = -I*dt/(4*dx*dx);
	cdouble c = -I*dt/(4*dx*dx);

	ws->alpha[0] = -a / (ws->b[0] + c);
	ws->betta[0] = ws->f[0] / (ws->b[0] + c);
	for (int i = 1; i < N; ++i) {
		ws->alpha[i] =                           - a / (ws->b[i] + c*ws->alpha[i-1]);
		ws->betta[i] = (ws->f[i] - c*ws->betta[i-1]) / (ws->b[i] + c*ws->alpha[i-1]);
	}

	psi[N-1] = ws->betta[N-1]/(1 - ws->alpha[N-1]);
	for (int i = N-2; i >= 0; --i) {
		psi[i] = ws->alpha[i]*psi[i+1] + ws->betta[i];
	}
}
