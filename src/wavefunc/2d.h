#pragma once

#include "../types.h"
#include "../grid.h"


struct Wavefunc2d {
	Grid2d const* grid;

	cdouble* data;
	bool data_own;

    Wavefunc2d(): grid(NULL), data(NULL), data_own(false) {}
    Wavefunc2d(Grid2d const* grid);
    Wavefunc2d(cdouble* data, Grid2d const* grid);
    ~Wavefunc2d();

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
    Wavefunc2d(cdouble* data, bool data_own, Grid2d const* grid);
};
