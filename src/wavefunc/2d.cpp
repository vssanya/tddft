#include "2d.h"

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
