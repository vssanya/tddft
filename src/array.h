#pragma once

#include "grid.h"

template <typename T, typename Grid, typename... Index>
class Array {
	public:
		Grid const grid;

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

		void mean(Array const* other) {
#pragma omp parallel for
			for (int i = 0; i < grid.size(); ++i) {
				data[i] = 0.5*(data[i] + other->data[i]);
			}
		}

		void add(Array const& other) {
#pragma omp parallel for
			for (int i = 0; i < grid.size(); ++i) {
				data[i] += other.data[i];
			}
		}

		void add_simd(Array const& other) {
#pragma omp parallel for simd
			for (int i = 0; i < grid.size(); ++i) {
				data[i] += other.data[i];
			}
		}

		void shift(int value, T fill_value) {
			assert(value > 0);

			for (int i=0; i<grid.size() - value;i++) {
				data[i] = data[i+value];
			}

			for (int i=grid.size() - value; i<grid.size(); i++) {
				data[i] = fill_value;
			}
		}

		int argmax(std::function<T(T)> func) {
			int argmax;
			T max = 0.0;
#pragma omp parallel
			{
				int local_argmax = 0;
				T local_max = 0.0;
#pragma omp for
				for (int i = 0; i < grid.size(); ++i) {
					T current = func(data[i]);
					if (local_max < current) {
						local_max = current;
						local_argmax = i;
					}
				}
#pragma omp critical
				{
					if (max < local_max) {
						max = local_max;
						argmax = local_argmax;
					}
				}
			}

			return argmax;
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
