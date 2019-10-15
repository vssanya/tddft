#pragma once

#include "grid.h"

template <typename T, typename Grid, typename... Index>
class Array {
	public:
		Grid const& grid;

		T* data;
		bool is_own;

		Array(): grid(Grid()), data(nullptr), is_own(false) {}

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

		Array(Array<T, Grid, Index...> const& arr): Array(arr.data, arr.grid) {}

		~Array() {
			if (is_own && data != nullptr) {
				delete[] data;
			}
		}

		void copy(Array* dest) const {
#pragma omp parallel for
			for (int i = 0; i < grid.size(); ++i) {
				dest->data[i] = data[i];
			}
		}

		void set(T value) const {
#pragma omp parallel for
			for (int i = 0; i < grid.size(); ++i) {
				data[i] = value;
			}
		}

		inline T& operator() (Index... index) {
			return data[grid.index(index...)];
		}

		inline T const& operator() (Index... index) const {
			return data[grid.index(index...)];
		}
};

template <typename T>
using Array1D = Array<T, Grid1d, int>;

template <typename T>
using Array2D = Array<T, Grid2d, int, int>;

template <typename T>
using Array3D = Array<T, Grid3d, int, int, int>;

template <typename T>
using ArraySp2D = Array<T, SpGrid2d, int, int>;

template <typename T>
using ArraySh = Array<T, ShGrid, int, int>;
