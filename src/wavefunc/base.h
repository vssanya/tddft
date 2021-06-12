#pragma once

#include "../types.h"
#include "../array.h"


template<typename Grid, typename... Index>
class WavefuncBase: public Array<cdouble, Grid, Index...> {
	public:
		WavefuncBase(cdouble* data, Grid const& grid): Array<cdouble, Grid, Index...>(data, grid) {}
		WavefuncBase(Grid const& grid): Array<cdouble, Grid, Index...>(grid) {}

		inline double abs_2(Index... index) const {
			cdouble const value = (*this)(index...);
			return pow(creal(value), 2) + pow(cimag(value), 2);
		}

		// \return \f$<\psi_1|\psi_2>\f$
		cdouble operator*(WavefuncBase const& other) const {
			return this->grid.template integrate<cdouble>([this, &other](Index... index) -> cdouble {
					return (*this)(index...)*conj(other(index...));
					}, typename Grid::Range(this->grid, other.grid));
		}

		void exclude(WavefuncBase const& other) {
			auto proj = (*this)*other / other.norm();

#pragma omp parallel for
			for (int i = 0; i < this->grid.size(); i++) {
				this->data[i] -= other.data[i]*proj;
			}
		}

		double norm(std::function<double (Index... index)> mask = nullptr) const {
			if (mask == nullptr) {
				return this->grid.template integrate<double>([this](Index... index) -> double {
						return abs_2(index...);
						});
			} else {
				return this->grid.template integrate<double>([this, mask](Index... index) -> double {
						return abs_2(index...)*mask(index...);
						});
			}
		}

		void normalize(double norm = 1.0) {
			double current_norm = this->norm();

#pragma omp parallel for
			for (int i = 0; i < this->grid.size(); ++i) {
				this->data[i] /= sqrt(current_norm/norm);
			}
		}
};

template <typename Grid>
using WavefuncBase1D = WavefuncBase<Grid, int>;

template <typename Grid>
using WavefuncBase2D = WavefuncBase<Grid, int, int>;

template <typename Grid>
using WavefuncBase3D = WavefuncBase<Grid, int, int, int>;
