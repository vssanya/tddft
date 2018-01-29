#include "wf_with_source.h"

void workspace::wf_E_with_source::prop_src(sh_wavefunc_t& wf, field_t const* field, double t, double dt) {
	double E = field_E(field, t + dt/2);

	for (int il=1; il<wf_source.grid->n[iL]; il++) {
#pragma omp parallel for
		for (int ir=0; ir<wf.grid->n[iR]; ir++) {
			double r = sh_grid_r(wf.grid, ir);

			wf(ir, il) += -I*r*E*clm(il-1, wf_source.m)*wf_source(ir, il-1)*dt;
		}
	}

	for (int il=0; il<wf_source.grid->n[iL]-1; il++) {
#pragma omp parallel for
		for (int ir=0; ir<wf.grid->n[iR]; ir++) {
			double r = sh_grid_r(wf.grid, ir);

			wf(ir, il) += -I*r*E*clm(il, wf_source.m)*wf_source(ir, il+1)*dt;
		}
	}
}
