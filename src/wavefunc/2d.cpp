#include "2d.h"
#include "cartesian_2d.h"

#include <iostream>


wavefunc_2d_t::wavefunc_2d_t(cdouble* data, bool data_own, grid2_t const* grid):
	grid(grid),
	data(data),
	data_own(data_own)
{
}

wavefunc_2d_t::wavefunc_2d_t(cdouble* data, grid2_t const* grid): wavefunc_2d_t(data, false, grid) {
}

wavefunc_2d_t::wavefunc_2d_t(grid2_t const* grid): wavefunc_2d_t(NULL, false, grid) {
	std::cout << "Call constructor";
	data = new cdouble[grid2_size(grid)]();
}

wavefunc_2d_t::~wavefunc_2d_t() {
	if (data_own) {
		delete[] data;
	}
}

double ct_wavefunc_t::norm() const {
	double norm = 0.0;

#pragma omp parallel for reduction(+:norm) collapse(2)
	for (int ir=0; ir<grid->n[iX]; ++ir) {
		for (int ic=0; ic<grid->n[iY]; ++ic) {
			double p = sp2_grid_r(grid, ir);

			norm += pow(cabs((*this)(ir,ic))*p, 2);
		}
	}

	return norm*grid->d[iX]*grid->d[iY];
}
