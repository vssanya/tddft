#include "wf_with_source.h"

void workspace::WfEWithSource::prop_src(ShWavefunc& wf, field_t const* field, double t, double dt) {
	double E = field_E(field, t + dt/2);

	cdouble c = -I*E*cexp(-I*source_E*(t+dt/2))*dt;

	for (int il=1; il<std::min(wf_source.grid->n[iL]+1, wf.grid->n[iL]); il++) {
#pragma omp parallel for
		for (int ir=0; ir<wf.grid->n[iR]; ir++) {
            double r = wf.grid->r(ir);

			wf(ir, il) += c*r*clm(il-1, wf_source.m)*wf_source(ir, il-1);
		}
	}

	for (int il=0; il<wf_source.grid->n[iL]-1; il++) {
#pragma omp parallel for
		for (int ir=0; ir<wf.grid->n[iR]; ir++) {
            double r = wf.grid->r(ir);

			wf(ir, il) += c*r*clm(il, wf_source.m)*wf_source(ir, il+1);
		}
	}
}


void workspace::WfEWithSource::prop_abs(ShWavefunc& wf, double dt) {
	assert(wf.grid->n[iR] == grid->n[iR]);
	assert(wf.grid->n[iL] <= grid->n[iL]);
	double norm = 0.0;
#pragma omp parallel for collapse(2) reduction(+:norm)
	for (int il = 0; il < wf.grid->n[iL]; ++il) {
		for (int ir = 0; ir < wf.grid->n[iR]; ++ir) {
			double c = exp(-uabs->u(ir)*dt);

			norm += wf.abs_2(ir, il)*pow(1-c, 2);
            wf(ir, il) *= c;
		}
	}

	abs_norm += norm*wf.grid->d[iR];
}
