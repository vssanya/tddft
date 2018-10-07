#include "wf_with_polarization.h"


void workspace::WfWithPolarization::prop(ShWavefunc& wf, field_t const* field, double t, double dt) {
	double Et = field_E(field, t + dt/2);

	sh_f Ul[2] = {
            [this](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
                return l*(l+1)/(2*r*r) + atom_cache->u(ir);
			},
            [this, Et](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
				return (r + Upol[ir])*Et*clm(l,m);
			}
	};


    prop_common(wf, dt, 2, Ul);

	prop_abs(wf, dt);
}
