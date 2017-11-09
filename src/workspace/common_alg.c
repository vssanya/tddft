#include "common_alg.h"


void wf_prop_ang_l(sh_wavefunc_t* wf, cdouble dt, int l, int l1, sh_f Ul) {
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

