#pragma once

#include <functional>

#include "types.h"
#include "utils.h"
#include "grid.h"
#include "sphere_harmonics.h"

#include "array.h"
#include "wavefunc/base.h"

#ifndef __CUDACC__
#include <optional>
#endif


/*!
 * \brief Волновая функция представленная в виде разложения по сферическим гармоникам
 *
 * \f[\psi(\vec{r}) = \frac{1}{r} \sum_{l=0}^{\infty} \Theta_{lm}(r) Y_{lm}(\theta, \phi) \simeq \frac{1}{r} \sum_{l=0}^{N_l - 1} \Theta_{lm}(r) Y_{lm}(\theta, \phi)\f]
 *
 * */
template<class Grid>
class Wavefunc: public WavefuncBase2D<Grid> {
	public:
		int m;         //!< is magnetic quantum number

		typedef double  (*func_wf_t        )(Wavefunc const* wf, int ir, int il);
		typedef cdouble (*func_complex_wf_t)(Wavefunc const* wf, int ir, int il);

		typedef std::function<double(int ir, int il, int m)> sh_f;

		template<class T>
			inline T integrate(std::function<T(int, int)> func, int l_max = -1, int l_min = 0) const {
				if (l_max == -1) {
					l_max = this->grid.n[iL] - 1;
				}

				l_min = std::max(l_min, m);

				T res = 0.0;
#pragma omp parallel for reduction(+:res) collapse(2)
				for (int il = l_min; il < l_max; ++il) {
					for (int ir = 0; ir < this->grid.n[iR]; ++ir) {
						res += func(ir, il)*this->grid.J(ir, il);
					}
				}
				return res*this->grid.d[iR];
			}

		Wavefunc(cdouble* data, Grid const& grid, int const m):
			WavefuncBase2D<Grid>(data, grid),
			m(m) {}

		Wavefunc(Grid const& grid, int const m): Wavefunc(nullptr, grid, m) {}
		~Wavefunc() {}

		using WavefuncBase2D<Grid>::operator();
		inline Array1D<cdouble> operator() (int il) {
			return Array1D<cdouble>(&(*this)(0, il), Grid1d(this->grid.n[iR]));
		}

		void init_p(double p, int l_max=-1) {
			if (l_max == -1) {
				l_max = this->grid.n[iL];
			}

			const cdouble i = {0.0, 1.0};

#pragma omp parallel for collapse(2)
			for (int l=0; l<l_max; l++) {
				for (int ir=0; ir<this->grid.n[iR]; ir++) {
					double r = this->grid.r(ir);
					(*this)(ir, l) = cpow(i, l)*sqrt(2/M_PI)*r*JlCache::calc(p*r, l);
				}
			}
		}

		inline cdouble d_dr(int ir, int il) const {
			return (-(*this)(ir+2, il) + 8*(*this)(ir+1, il) - 8*(*this)(ir-1, il) + (*this)(ir-2, il))/(12*this->grid.d[iR]);
		}

		inline cdouble d_dr_save(int ir, int il) const {
			if (ir == 0) {
				return (-(*this)(ir+2, il) + 8*(*this)(ir+1, il))/(12*this->grid.d[iR]);
			} else if (ir == 1) {
				return (-(*this)(ir+2, il) + 8*(*this)(ir+1, il) - 8*(*this)(ir-1, il))/(12*this->grid.d[iR]);
			} else {
				return d_dr(ir, il);
			}
		}

		// \return \f$<\psi_1|\psi_2>\f$
		cdouble operator*(Wavefunc const& other) const {
			return integrate<cdouble>([this, &other](int ir, int il) -> cdouble {
					return (*this)(ir, il)*conj(other(ir, il));
					}, std::min(this->grid.n[iL], other.grid.n[iL]));
		}

#ifndef __CUDACC__
		void sh_prod(Wavefunc const& other, Wavefunc& res, std::optional<int> lmax = std::nullopt) const {
			res.m = 0;

			int Lmax = lmax.value_or(this->grid.N[iL]);

			for (int ir = 0; ir < res.grid.N[iR]; ir++) {
				for (int l = 0; l < res.grid.N[iL]; l++) {
					res(ir, l) = 0.0;

					for (int l1 = 0; l1 < Lmax; l1++) {
						for (int l2 = std::abs(l-l1); l2 < std::min(std::abs(l+l1), Lmax); l2++) {
							res(ir, l) += std::pow(-1, other.m) * std::sqrt((2*l1 + 1)/(2*l2 + 1)) *
								clebsch_gordan_coef(l1, 0, l, 0, l2, 0) *
								clebsch_gordan_coef(l1, this->m, l, res.m, l2, -other.m) *
								(*this)(ir, l1) * std::conj(other(ir, l2));
						}
					}

					res(ir, l) *= std::sqrt(2*l + 1) / std::sqrt(4*M_PI);
				}
			}
		}

		// (1 + x)^n = 1 + a1*x + a2*x
		void pow_n(Wavefunc& res, double n, double a1, double a2) {
			for (int ir = 0; ir < res.grid.N[iR]; ir++) {
				auto p00 = (*this)(ir, 0) / std::sqrt(4*M_PI);

				for (int l = 0; l < res.grid.N[iL]; l++) {
					if (l == 0) {
						res(ir, l) = 1.0;
					} else {
						res(ir, l) = a1*(*this)(ir, l) / p00;
					}

					//cdouble tmp = 0.0;
					//for (int l1 = 0; l1 < this->grid.N[iL]; l1++) {
						//for (int l2 = std::abs(l-l1); l2 < std::min(std::abs(l+l1), this->grid.N[iL]); l2++) {
							//tmp += std::sqrt((2*l1 + 1)*(2*l2 + 1)) *
								//clebsch_gordan_coef(l1, 0, l, 0, l2, 0) *
								//clebsch_gordan_coef(l1, this->m, l, res.m, l2, -other.m) *
								//(*this)(ir, l1) * std::conj(other(ir, l2));
						//}
					//}

					res(ir, l) *= pow(p00, n);
				}
			}
		}

		void pow_1_3(Wavefunc& res) const {
		}
#endif

		void exclude(Wavefunc const& other) {
			auto proj = (*this)*other / other.norm();

#pragma omp parallel for collapse(2)
			for (int il = this->m; il < this->grid.n[iL]; il++) {
				for (int ir = 0; ir < this->grid.n[iR]; ir++) {
					(*this)(ir, il) -= other(ir, il)*proj;
				}
			}
		}

		// <psi|U(r)cos(\theta)|psi>
		double cos(sh_f func) const {
			return 2*integrate<double>([this, func](int ir, int il) -> double {
					return clm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+1)))*func(ir, il, m);
					}, this->grid.n[iL]-1, std::abs(m));
		}

		cdouble cos(sh_f func, Wavefunc const& other, int l_max=-1) const {
			return integrate<cdouble>([this, func, &other](int ir, int il) -> cdouble {
					return clm(il, m)*((*this)(ir, il)*conj(other(ir, il+1)) + (*this)(ir, il+1)*conj(other(ir, il)))*func(ir, il, m);
					}, l_max, std::abs(m));
		}

		// <psi|U(r)cos^2(\theta)|psi>
		double cos2(sh_f func) const {
			double res = 0.0;
			res += integrate<double>([this, func](int ir, int il) -> double {
					cdouble psi = (*this)(ir, il);
					return (plm(il, m) + 0.5)*abs2(psi)*func(ir, il, m);
					}, this->grid.n[iL]);

			res += 2*integrate<double>([this, func](int ir, int il) -> double {
					return qlm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+2)))*func(ir, il, m);
					}, this->grid.n[iL]-2);

			return res*(2.0/3.0);
		}

		double proj_Y10(sh_f func) const {
			return 2*0.5*sqrt(3/M_PI)*integrate<double>([this, func](int ir, int l) -> double {
					return clm(l, m)*creal((*this)(ir, l)*conj((*this)(ir, l+1)))*func(ir, l, m);
					}, this->grid.n[iL]-1, 1);
		}

		double proj_Y30(sh_f func) const {
			double res = 0.0;

			res += 3*integrate<double>([this, func](int ir, int l) -> double {
					return clm(l, m)*double(l*l + 2*l - 5*m*m)/double((2*l - 1)*(2*l + 5))*
					creal((*this)(ir, l)*conj((*this)(ir, l+1)))*func(ir, l, m);
					}, this->grid.n[iL]-1, 1);

			res += 5*integrate<double>([this, func](int ir, int l) -> double {
					return clm(l,m)*clm(l+1,m)*clm(l+2,m)*
					creal((*this)(ir, l)*conj((*this)(ir, l+3)))*func(ir, l, m);
					}, this->grid.n[iL]-3, 0);

			return res*2*0.25*sqrt(7/M_PI);
		}

		double cos3(sh_f func) const {
			return 0.2*sqrt(M_PI)*(4.0/sqrt(7.0)*proj_Y30(func) + 2*sqrt(3)*proj_Y10(func));
		}

		// <psi|U(r)sin^2(\theta)|psi>
		double sin2(sh_f func) const {
			double res = 0.0;
			res += integrate<double>([this, func](int ir, int il) -> double {
					cdouble psi = (*this)(ir, il);
					return (0.5 - plm(il, m))*(creal(psi)*creal(psi) + cimag(psi)*cimag(psi))*func(ir, il, m);
					}, this->grid.n[iL]);

			res -= 2*integrate<double>([this, func](int ir, int il) -> double {
					return qlm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+2)))*func(ir, il, m);
					}, this->grid.n[iL]-2);

			return res*(2.0/3.0);
		}

		void cos_r(sh_f U, double* res) const {
			for (int ir = 0; ir < this->grid.n[iR]; ++ir) {
				res[ir] = 0.0;
				for (int il = this->m; il < this->grid.n[iL]-1; ++il) {
					res[ir] += 2*clm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+1)))*U(ir, il, m);
				}
			}
		}

		double cos_r2(sh_f U, int Z) const {
			return 2*integrate<double>([this, U](int ir, int il) -> double {
					return clm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+1)))*U(ir, il, m);
					}, this->grid.n[iL]-1);
		}

		double norm(sh_f mask = nullptr) const {
			if (mask == nullptr) {
				return  integrate<double>([this](int ir, int il) {
						return this->abs_2(ir, il);
						}, this->grid.n[iL]);
			} else {
				return integrate<double>([this, mask](int ir, int il) {
						return this->abs_2(ir, il)*mask(ir, il, m);
						}, this->grid.n[iL]);
			}
		}

#ifndef __CUDACC__
		// res(z) = int <psi|psi> dx dy
		void norm_z(double* res) const {
			int Nr = this->grid.n[iR];

			double f[Nr];
			double dp[Nr];

			for (int iz = 0; iz < Nr-1; ++iz) {
				double z = this->grid.r(iz);

				double rho_prev = 0.0;
				f[iz] = abs2(get_sp(iz, 0.0))*0.0;

				for (int ir = iz+1; ir < Nr; ++ir) {
					double r = this->grid.r(ir);
					double theta = acos(z/r);

					double rho = r*sin(theta);

					f[ir] = abs2(get_sp(ir, theta))*rho;
					dp[ir-1] = rho - rho_prev;

					rho_prev = rho;
				}

				res[Nr + iz] = integrate_1d_trap(&f[iz], &dp[iz], Nr - iz);
			}

			res[2*Nr-1] = 0.0;
			res[0] = 0.0;

			for (int iz = 0; iz < Nr-1; ++iz) {
				double z = - this->grid.r(iz);

				double rho_prev = 0.0;
				f[iz] = abs2(get_sp(iz, 0.0))*0.0;

				for (int ir = iz+1; ir < Nr; ++ir) {
					double r = this->grid.r(ir);
					double theta = acos(z/r);

					double rho = r*sin(theta);

					f[ir] = abs2(get_sp(ir, theta))*rho;
					dp[ir-1] = rho - rho_prev;

					rho_prev = rho;
				}

				res[Nr-1-iz] = integrate_1d_trap<double>(&f[iz], &dp[iz], Nr - iz);
			}
		}
#endif

		void normalize(double norm = 1.0) {
			double current_norm = this->norm();
#pragma omp parallel for
			for (int i = 0; i < this->grid.size(); ++i) {
				this->data[i] /= sqrt(current_norm/norm);
			}
		}

		double z(sh_f mask = nullptr) const {
			if (mask == nullptr) {
				return cos([this](int ir, int il, int im) {
						return this->grid.r(ir);
						});
			} else {
				return cos([this, &mask](int ir, int il, int im) {
						return this->grid.r(ir)*mask(ir, il, im);
						});
			}
		}

		double z2(sh_f mask = nullptr) const {
			if (mask == nullptr) {
				return cos2([this](int ir, int il, int im) {
						return pow(this->grid.r(ir), 2);
						});
			} else {
				return cos2([this, &mask](int ir, int il, int im) {
						return pow(this->grid.r(ir), 2)*mask(ir, il, im);
						});
			}
		}

		cdouble pz() const {
			cdouble i = {0.0, 1.0};
			return i*integrate<cdouble>([this](int ir, int il) -> cdouble {
					double r = this->grid.r(ir);
					cdouble psi_0 = conj((*this)(ir, il));
					cdouble psi_1 = (*this)(ir, il-1);

					return -clm(il-1, m)*psi_0*(
							d_dr_save(ir, il-1) -
							il*psi_1/r);
					}, this->grid.n[iL], 1) +
			i*integrate<cdouble>([this](int ir, int il) -> cdouble {
					double r = this->grid.r(ir);
					cdouble psi_0 = conj((*this)(ir, il));
					cdouble psi_1 = (*this)(ir, il+1);

					return -clm(il, m)*psi_0*(
							d_dr_save(ir, il+1)
							+ (il + 1)*psi_1/r);
					}, this->grid.n[iL] - 1, 0);
		}

		void random_l(int l) {
			assert(l >= 0 && l < this->grid.n[iL]);

			for (int il=0; il<this->grid.n[iL]; ++il) {
				for (int ir=0; ir<this->grid.n[iR]; ++ir) {
					(*this)(ir, il) = 0.0;
				}
			}

			{
				int il = l;
				cdouble* psi = &(*this)(0, il);

				for (int ir=0; ir<this->grid.n[iR]; ++ir) {
					double const r = this->grid.r(ir);
					psi[ir] = (double)rand()/(double)RAND_MAX*r*exp(-r/(2*l+1));
				}

				for (int i=0; i<10; ++i) {
					for (int ir=1; ir<this->grid.n[iR]-1; ++ir) {
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

				norm[in] = wfs[in]->norm();
			}
		}


#ifndef __CUDACC__
		/*!
		 * \return \f$\psi(r, \Omega)\f$
		 * */
		cdouble get_sp(SpGrid const& grid, int i[3], YlmCache const* ylm_cache, std::optional<int> Lmax = std::nullopt) const {
			cdouble res = 0.0;
			int L = std::min(Lmax.value_or(this->grid.n[iL]), this->grid.n[iL]);

			for (int l = 0; l < L; ++l) {
				res += (*this)(i[iR], l)*(*ylm_cache)(l, m, i[iT]);
			}

			double r = this->grid.r(i[iR]);
			return res/r;
		}

		cdouble get_sp(int ir, double theta) const {
			cdouble res = 0.0;
			for (int l = 0; l < this->grid.n[iL]; ++l) {
				res += (*this)(ir, l)*YlmCache::calc(l, m, theta);
			}

			double r = this->grid.r(ir);
			return res/r;
		}

		void n_sp(SpGrid const& grid, double* n, YlmCache const* ylm_cache, std::optional<int> Lmax = std::nullopt) const {
#pragma omp parallel for collapse(2)
			for (int ir = 0; ir < grid.n[iR]; ++ir) {
				for (int it = 0; it < grid.n[iT]; ++it) {
					int index[3] = {ir, it, 0};
					cdouble const psi = get_sp(grid, index, ylm_cache, Lmax);
					n[ir + it*grid.n[iR]] = abs2(psi);
				}
			}
		}
#endif

		void n_l0(double* n) const {
#pragma omp parallel for
			for (int ir = 0; ir < this->grid.n[iR]; ++ir) {
				double res = 0.0;
				for (int il = this->m; il < this->grid.n[iL]; ++il) {
					res += this->abs_2(ir, il);
				}
				n[ir] = res / (pow(this->grid.r(ir), 2)*4*M_PI);
			}
		}
};

typedef Wavefunc<ShGrid> ShWavefunc;
typedef Wavefunc<ShNotEqudistantGrid> ShNeWavefunc;
typedef Wavefunc<SpGrid2d> SpWavefunc2d;
