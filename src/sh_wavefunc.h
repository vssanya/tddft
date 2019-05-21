#pragma once

#include <stdbool.h>
#include <math.h>
#include <functional>

#include "types.h"
#include "utils.h"
#include "grid.h"
#include "sphere_harmonics.h"

#include "array.h"

/*!
 * \brief Волновая функция представленная в виде разложения по сферическим гармоникам
 *
 * \f[\psi(\vec{r}) = \frac{1}{r} \sum_{l=0}^{\infty} \Theta_{lm}(r) Y_{lm}(\theta, \phi) \simeq \frac{1}{r} \sum_{l=0}^{N_l - 1} \Theta_{lm}(r) Y_{lm}(\theta, \phi)\f]
 *
 * */
template<class Grid>
class Wavefunc {
	public:
		Grid const& grid;

		cdouble* data; //!< data[i + l*grid.Nr] = \f$\Theta_{lm}(r_i)\f$
		bool data_own; //!< кто выделил данные

		int m;         //!< is magnetic quantum number

		typedef double  (*func_wf_t        )(Wavefunc const* wf, int ir, int il);
		typedef cdouble (*func_complex_wf_t)(Wavefunc const* wf, int ir, int il);

		typedef std::function<double(int ir, int il, int m)> sh_f;

		template<class T>
			inline T integrate(std::function<T(int, int)> func, int l_max = -1, int l_min = 0) const {
				if (l_max == -1) {
					l_max = grid.n[iL] - 1;
				}

				T res = 0.0;
#pragma omp parallel for reduction(+:res) collapse(2)
				for (int il = l_min; il < l_max; ++il) {
					for (int ir = 0; ir < grid.n[iR]; ++ir) {
						res += func(ir, il)*grid.J(ir, il);
					}
				}
				return res*grid.d[iR];
			}

		Wavefunc(cdouble* data, Grid const& grid, int const m):
			data(data),
			grid(grid),
			m(m),
			data_own(false) {
				if (data == nullptr) {
					this->data = new cdouble[grid.size()]();
					data_own = true;
				}
			}

		Wavefunc(Grid const& grid, int const m): Wavefunc(nullptr, grid, m) {}

		~Wavefunc() {
			if (data_own) {
				delete[] data;
			}
		}

		inline Array1D<cdouble> operator() (int il) {
			return Array1D<cdouble>(&(*this)(0, il), Grid1d(grid.n[iR]));
		}

		inline cdouble& operator() (int ir, int il) {
			assert(ir < grid.n[iR] && il < grid.n[iL]);
			return data[ir + il*grid.n[iR]];
		}

		inline cdouble const& operator() (int ir, int il) const {
			assert(ir < grid.n[iR] && il < grid.n[iL]);
			return data[ir + il*grid.n[iR]];
		}

		inline cdouble d_dr(int ir, int il) const {
			return (-(*this)(ir+2, il) + 8*(*this)(ir+1, il) - 8*(*this)(ir-1, il) + (*this)(ir-2, il))/(12*grid.d[iR]);
		}

		inline cdouble d_dr_save(int ir, int il) const {
			if (ir == 0) {
				return (-(*this)(ir+2, il) + 8*(*this)(ir+1, il))/(12*grid.d[iR]);
			} else if (ir == 1) {
				return (-(*this)(ir+2, il) + 8*(*this)(ir+1, il) - 8*(*this)(ir-1, il))/(12*grid.d[iR]);
			} else {
				return d_dr(ir, il);
			}
		}

		inline double abs_2(int ir, int il) const {
			cdouble const value = (*this)(ir, il);
			return pow(creal(value), 2) + pow(cimag(value), 2);
		}

		void copy(Wavefunc* wf_dest) const {
#pragma omp parallel for
			for (int i = 0; i < grid.size(); ++i) {
				wf_dest->data[i] = data[i];
			}
		}

		// \return \f$<\psi_1|\psi_2>\f$
		cdouble operator*(Wavefunc const& other) const {
			return integrate<cdouble>([this, &other](int ir, int il) -> cdouble {
					return (*this)(ir, il)*conj(other(ir, il));
					}, std::min(grid.n[iL], other.grid.n[iL]));
		}

		void exclude(Wavefunc const& other) {
			auto proj = (*this)*other / other.norm();

#pragma omp parallel for collapse(2)
			for (int il = 0; il < grid.n[iL]; il++) {
				for (int ir = 0; ir < grid.n[iR]; ir++) {
					(*this)(ir, il) -= other(ir, il)*proj;
				}
			}
		}

		// <psi|U(r)cos(\theta)|psi>
		double cos(sh_f func) const {
			return 2*integrate<double>([this, func](int ir, int il) -> double {
					return clm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+1)))*func(ir, il, m);
					}, grid.n[iL]-1);
		}

		cdouble cos(sh_f func, Wavefunc const& other, int l_max=-1) const {
			return integrate<cdouble>([this, func, &other](int ir, int il) -> cdouble {
					return clm(il, m)*((*this)(ir, il)*conj(other(ir, il+1)) + (*this)(ir, il+1)*conj(other(ir, il)))*func(ir, il, m);
					}, l_max);
		}

		// <psi|U(r)cos^2(\theta)|psi>
		double cos2(sh_f func) const {
			double res = 0.0;
			res += integrate<double>([this, func](int ir, int il) -> double {
					cdouble psi = (*this)(ir, il);
					return (plm(il, m) + 0.5)*(creal(psi)*creal(psi) + cimag(psi)*cimag(psi))*func(ir, il, m);
					}, grid.n[iL]);

			res += 2*integrate<double>([this, func](int ir, int il) -> double {
					return qlm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+2)))*func(ir, il, m);
					}, grid.n[iL]-2);

			return res*(2.0/3.0);
		}

		// <psi|U(r)sin^2(\theta)|psi>
		double sin2(sh_f func) const {
			double res = 0.0;
			res += integrate<double>([this, func](int ir, int il) -> double {
					cdouble psi = (*this)(ir, il);
					return (0.5 - plm(il, m))*(creal(psi)*creal(psi) + cimag(psi)*cimag(psi))*func(ir, il, m);
					}, grid.n[iL]);

			res -= 2*integrate<double>([this, func](int ir, int il) -> double {
					return qlm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+2)))*func(ir, il, m);
					}, grid.n[iL]-2);

			return res*(2.0/3.0);
		}

		void cos_r(sh_f U, double* res) const {
			for (int ir = 0; ir < grid.n[iR]; ++ir) {
				res[ir] = 0.0;
				for (int il = 0; il < grid.n[iL]-1; ++il) {
					res[ir] += 2*clm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+1)))*U(ir, il, m);
				}
			}
		}

		double cos_r2(sh_f U, int Z) const {
			return 2*integrate<double>([this, U](int ir, int il) -> double {
					return clm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+1)))*U(ir, il, m);
					}, grid.n[iL]-1);
		}

		double norm(sh_f mask = nullptr) const {
			if (mask == nullptr) {
				return  integrate<double>([this](int ir, int il) {
						return abs_2(ir, il);
						}, grid.n[iL]);
			} else {
				return integrate<double>([this, mask](int ir, int il) {
						return abs_2(ir, il)*mask(ir, il, m);
						}, grid.n[iL]);
			}
		}

		void normalize() {
			double norm = this->norm(NULL);
#pragma omp parallel for
			for (int i = 0; i < grid.size(); ++i) {
				data[i] /= sqrt(norm);
			}
		}

		double z(sh_f mask = nullptr) const {
			if (mask == nullptr) {
				return cos([this](int ir, int il, int im) {
						return grid.r(ir);
						});
			} else {
				return cos([this, &mask](int ir, int il, int im) {
						return grid.r(ir)*mask(ir, il, im);
						});
			}
		}

		cdouble pz() const {
			cdouble i = {0.0, 1.0};
			return i*integrate<cdouble>([this](int ir, int il) -> cdouble {
					double r = grid.r(ir);
					cdouble psi_0 = conj((*this)(ir, il));
					cdouble psi_1 = (*this)(ir, il-1);

					return -clm(il-1, m)*psi_0*(
							d_dr_save(ir, il-1) -
							il*psi_1/r);
					}, grid.n[iL], 1) +
			i*integrate<cdouble>([this](int ir, int il) -> cdouble {
					double r = grid.r(ir);
					cdouble psi_0 = conj((*this)(ir, il));
					cdouble psi_1 = (*this)(ir, il+1);

					return -clm(il, m)*psi_0*(
							d_dr_save(ir, il+1)
							+ (il + 1)*psi_1/r);
					}, grid.n[iL] - 1, 0);
		}

		void random_l(int l) {
			assert(l >= 0 && l < grid.n[iL]);

			for (int il=0; il<grid.n[iL]; ++il) {
				for (int ir=0; ir<grid.n[iR]; ++ir) {
					(*this)(ir, il) = 0.0;
				}
			}

			{
				int il = l;
				cdouble* psi = &(*this)(0, il);

				for (int ir=0; ir<grid.n[iR]; ++ir) {
					double const r = grid.r(ir);
					psi[ir] = (double)rand()/(double)RAND_MAX*r*exp(-r/(2*l+1));
				}

				for (int i=0; i<10; ++i) {
					for (int ir=1; ir<grid.n[iR]-1; ++ir) {
						psi[ir] = (psi[ir-1] + psi[ir] + psi[ir+1])/3.0;
					}
				}
			}
		}

		static void ort_l(int l, int n, Wavefunc** wfs) {
			assert(n > 1);
			for (int in=1; in<n; ++in) {
				assert(wfs[in-1]->m == wfs[in]->m);
			}

			auto& grid = wfs[0]->grid;

			cdouble proj[n];
			double norm[n];

			for (int in=0; in<n; ++in) {
				for (int ip=0; ip<in; ++ip) {
					if (norm[ip] == 0.0) {
						proj[ip] = 0.0;
					} else {
						proj[ip] = ((*wfs[ip])*(*wfs[in])) / norm[ip];
					}
				}

				for (int ip=0; ip<in; ++ip) {
					cdouble* psi = &(*wfs[in])(0, l);
					for (int ir=0; ir<grid.n[iR]; ++ir) {
						psi[ir] -= proj[ip]*(*wfs[ip])(ir, l);
					}
				}

				norm[in] = wfs[in]->norm(NULL);
			}
		}

		/*!
		 * \return \f$\psi(r, \Omega)\f$
		 * */
		cdouble get_sp(SpGrid const& grid, int i[3], YlmCache const* ylm_cache) const {
			cdouble res = 0.0;
			for (int il = 0; il < this->grid.n[iL]; ++il) {
				int const l = this->grid.l(il);
				res += (*this)(i[iR], il)*(*ylm_cache)(l, m, i[iC]);
			}

			double r = this->grid.r(i[iR]);
			return res/r;
		}

		void n_sp(SpGrid const& grid, double* n, YlmCache const* ylm_cache) const {
#pragma omp parallel for collapse(2)
			for (int ir = 0; ir < grid.n[iR]; ++ir) {
				for (int ic = 0; ic < grid.n[iC]; ++ic) {
					int index[3] = {ir, ic, 0};
					cdouble const psi = get_sp(grid, index, ylm_cache);
					n[ir + ic*grid.n[iR]] = pow(creal(psi), 2) + pow(cimag(psi), 2);
				}
			}
		}

		void print() const {
			for (int ir = 0; ir < grid.n[iR]; ++ir) {
				double const r = grid.r(ir);
				double res = 0.0;
				for (int il = 0; il < grid.n[iL]; ++il) {
					res += pow(cabs((*this)(ir, il)), 2);
				}
				printf("%f ", res/(r*r));
			}
			printf("\n");
		}
};

typedef Wavefunc<ShGrid> ShWavefunc;
typedef Wavefunc<ShNotEqudistantGrid> ShNeWavefunc;
