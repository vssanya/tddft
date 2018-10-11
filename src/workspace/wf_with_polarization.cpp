#include "wf_with_polarization.h"


void workspace::WfWithPolarization::prop(ShWavefunc& wf, field_t const* field, double t, double dt) {
	double Et = field_E(field, t + dt/2);

	sh_f Ul[3];
	int Ul_size = 2;

	Ul[1] = [this, Et](ShGrid const* grid, int ir, int l, int m) -> double {
		double const r = grid->r(ir);
		return (r + Upol_1[ir])*Et*clm(l,m);
	};

	if (Upol_2 == nullptr) {
		Ul[0] = [this](ShGrid const* grid, int ir, int l, int m) -> double {
			double const r = grid->r(ir);
			return l*(l+1)/(2*r*r) + atom_cache->u(ir);
		};
	} else {
		Ul[0] = [this](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
				return l*(l+1)/(2*r*r) + atom_cache->u(ir) + plm(l,m)*Upol_2[ir];
		};

		Ul[2] = [this](ShGrid const* grid, int ir, int l, int m) -> double {
			return qlm(l, m)*Upol_2[ir];
		};
	}

	prop_common(wf, dt, Ul_size, Ul);

	prop_abs(wf, dt);
}
