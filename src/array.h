#pragma once

#include "grid.h"

template <typename T, typename Grid, typename... Index>
class Array {
	public:
		Grid const& grid;

		T* data;
		bool is_own;

		Array(T* data, Grid const& grid):
			grid(grid),
			data(data),
			is_own(false) {
				if (data == nullptr) {
					this->data = new T[grid.size()]();
					is_own = true;
				}
			}

		Array(Grid const& grid): Array(nullptr, grid) {}

		~Array() {
			if (is_own) {
				delete[] data;
			}
		}

		void copy(Array* dest) const {
#pragma omp parallel for
			for (int i = 0; i < grid.size(); ++i) {
				dest->data[i] = data[i];
			}
		}


		inline T& operator() (Index... index) {
			return data[grid.index(index...)];
		}

		inline T const& operator() (Index... index) const {
			return data[grid.index(index...)];
		}
};

template <typename T, typename Grid>
using Array1D = Array<T, Grid, int>;

template <typename T, typename Grid>
using Array2D = Array<T, Grid, int, int>;

template <typename T, typename Grid>
using Array3D = Array<T, Grid, int, int, int>;
