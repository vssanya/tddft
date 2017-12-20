#include "sfa.h"
#include "math.h"


using namespace workspace::sfa;

cdouble V_0(double p[2], field_t const* field, double t) {
	double E = field_E(field, t);
	double A = field_A(field, t);

	return -32*I*sqrt(M_PI)*E*(p[0]+A)/pow(1 + pow(p[0]+A, 2) + p[1]*p[1], 3)*cexp(I*t/2);
}

void momentum_space::propagate(ct_wavefunc_t& wf, field_t const* field, double t, double dt) {
	double A_t = field_A(field, t);
	double A_t_dt = field_A(field, t+dt);

#pragma omp parallel for collapse(2)
	for (int ix=0; ix<wf.grid->n[iX]; ++ix) {
		for (int iy=0; iy<wf.grid->n[iY]; ++iy) {
			double p[2] = {ct_grid_x(wf.grid, ix), ct_grid_y(wf.grid, iy)};

			wf(ix, iy) = ((I - 0.25*dt*(p[1]*p[1] + pow(p[0] + A_t, 2)))*wf(ix, iy) + 0.5*dt*(V_0(p, field, t+dt) + V_0(p, field, t)))/(I + 0.25*dt*(p[1]*p[1] + pow(p[0] + A_t_dt, 2)));
		}
	}
}
