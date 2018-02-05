#pragma once

#include "../types.h"
#include "../grid.h"

#include "2d.h"


struct ct_wavefunc_t : public wavefunc_2d_t {
	ct_wavefunc_t(): wavefunc_2d_t() {}
	ct_wavefunc_t(grid2_t const* grid): wavefunc_2d_t(grid) {}

	double norm() const;
};
