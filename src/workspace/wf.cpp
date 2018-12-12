#include <stdlib.h>
#include <stdio.h>

#include "wf.h"
#include "common_alg.h"


void workspace::WfE::prop(ShWavefunc& wf, field_t const* field, double t, double dt) {
	double Et = field_E(field, t + dt/2);

	sh_f Ul[2] = {
            [this](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
                return l*(l+1)/(2*r*r) + atom_cache.u(ir);
			},
            [Et](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
				return r*Et*clm(l,m);
			}
	};


    prop_common(wf, dt, 2, Ul);
	prop_abs(wf, dt);
}

void workspace::WfA::prop(ShWavefunc& wf, field_t const* field, double t, double dt) {
	double At = -field_A(field, t + dt/2);

	sh_f Ul[1] = {
            [this](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
                return l*(l+1)/(2*r*r) + atom_cache.u(ir);
			},
	};

	sh_f Al[2] = {
        [At](ShGrid const* grid, int ir, int l, int m) -> double {
				return At*clm(l,m);
		},
        [At](ShGrid const* grid, int ir, int l, int m) -> double {
				double const r = grid->r(ir);
				return At*(l+1)*clm(l,m)/r;
		}
	};


    prop_common(wf, dt, 1, Ul, Al);
	prop_abs(wf, dt);
}
