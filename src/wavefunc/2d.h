#pragma once

#include "../types.h"
#include "../grid.h"


struct wavefunc_2d_t {
	grid2_t const* grid;

	cdouble* data;
	bool data_own;

	wavefunc_2d_t(): grid(NULL), data(NULL), data_own(false) {}
	wavefunc_2d_t(grid2_t const* grid);
	wavefunc_2d_t(cdouble* data, grid2_t const* grid);
	~wavefunc_2d_t();

	double norm() const;

	inline
		cdouble& operator() (int i, int j) {
			return data[i + j*grid->n[0]];
		}

	inline
		cdouble operator() (int i, int j) const {
			return data[i + j*grid->n[0]];
		}

	private:
	wavefunc_2d_t(cdouble* data, bool data_own, grid2_t const* grid);
};
