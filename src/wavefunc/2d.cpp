#include "2d.h"
#include "cartesian_2d.h"

#include <iostream>


Wavefunc2d::Wavefunc2d(cdouble* data, bool data_own, Grid2d const* grid):
	grid(grid),
	data(data),
	data_own(data_own)
{
}

Wavefunc2d::Wavefunc2d(cdouble* data, Grid2d const* grid): Wavefunc2d(data, false, grid) {
}

Wavefunc2d::Wavefunc2d(Grid2d const* grid): Wavefunc2d(NULL, false, grid) {
	data = new cdouble[grid->size()]();
}

Wavefunc2d::~Wavefunc2d() {
	if (data_own) {
		delete[] data;
	}
}

double CtWavefunc::norm() const {
	double norm = 0.0;
    auto sp_grid = static_cast<SpGrid2d const*>(this->grid);

#pragma omp parallel for reduction(+:norm) collapse(2)
    for (int ir=0; ir<sp_grid->n[iX]; ++ir) {
        for (int ic=0; ic<sp_grid->n[iY]; ++ic) {
            double p = sp_grid->r(ir);
			double theta = sp_grid->theta(ic);

			norm += pow(cabs((*this)(ir,ic))*p, 2)*sin(theta);
		}
	}

	return norm*sp_grid->d[iR]*sp_grid->d[iC];
}
