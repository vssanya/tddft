#include "sfa.h"
#include "math.h"


using namespace workspace::sfa;

cdouble V_0(double p[2], double E, double A, double t) {
	return -32*I*sqrt(M_PI)*E*(p[0]+A)/pow(1 + pow(p[0]+A, 2) + p[1]*p[1], 3)*cexp(I*t/2);
}

void momentum_space::propagate(CtWavefunc& wf, field_t const* field, double t, double dt) {
	double A_t = field_A(field, t);
	double A_t_dt = field_A(field, t+dt);

	double E_t = field_E(field, t);
	double E_t_dt = field_E(field, t+dt);

    auto grid = *static_cast<SpGrid2d const*>(wf.grid);

#pragma omp parallel for collapse(2)
    for (int ir=0; ir<grid.n[iX]; ++ir) {
        for (int ic=0; ic<grid.n[iY]; ++ic) {
            double mod_p = grid.r(ir);
            double px = grid.c(ic)*mod_p;
			double py = sqrt(mod_p*mod_p - px*px);

			double p[2] = {px, py};

			wf(ir, ic) = ((I + 0.25*dt*(p[1]*p[1] + pow(p[0] + A_t, 2)))*wf(ir, ic) + 0.5*dt*(V_0(p, E_t_dt, A_t_dt, t+dt) + V_0(p, E_t, A_t, t)))/(I - 0.25*dt*(p[1]*p[1] + pow(p[0] + A_t_dt, 2)));
		}
	}
}
