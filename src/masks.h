#pragma once
#include "grid.h"
#include "utils.h"

// Маска для расчета нормы вблизи ядра
class CoreMask {
	public:
		double r_core;
		double dr;
	
		CoreMask(double r_core, double dr): r_core(r_core), dr(dr) {}

	double operator()(ShGrid const* grid, int ir, int il, int im) const {
		double const r = grid->r(ir);
		return smoothstep(r_core + dr - r, 0.0, 2*dr);  
	}
};
