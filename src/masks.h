#pragma once
#include "grid.h"
#include "utils.h"

// Маска для расчета нормы вблизи ядра
template <typename Grid>
class CoreMask {
	public:
		Grid const* grid;

		double r_core;
		double dr;
	
		CoreMask(Grid const* grid, double r_core, double dr): grid(grid), r_core(r_core), dr(dr) {}

	double operator()(int ir, int il, int im) const {
		double const r = grid->r(ir);
		return smoothstep(r_core + dr - r, 0.0, 2*dr);  
	}
};
