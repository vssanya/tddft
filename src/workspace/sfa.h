#pragma once

#include "../wavefunc/cartesian_2d.h"
#include "../fields.h"


namespace workspace {
	namespace sfa {
		class momentum_space {
			public:
				void propagate(ct_wavefunc_t& wf, field_t const* field, double t, double dt);
		};
	}
}
