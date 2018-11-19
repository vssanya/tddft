#pragma once

#include "grid.h"

class Array2D {
	public:
		Array2D(Grid2d const& grid): grid(grid), is_own(true) {
			data = new double[grid.size()];
		}

		Array2D(Grid2d const& grid, double* data): grid(grid), data(data), is_own(false) {
		}

		~Array2D() {
			if (is_own) {
				delete[] data;
			}
		}

		inline double& operator() (int ix, int iy) {
			assert(ix < grid->n[iX] && iy < grid->n[iY]);
			return data[ix + iy*grid.n[iX]];
		}

		inline double const& operator() (int ix, int iy) const {
			assert(ix < grid->n[iX] && iy < grid->n[iY]);
			return data[ix + iy*grid.n[iX]];
		}
	
		Grid2d const& grid;

		double* data;
		bool is_own;
};
