#pragma once

#include <functional>
#ifndef __CUDACC__
#include <optional>
#endif

#include "grid.h"


class JlCache {
public:
	double* data;

	int l_max;
	SpGrid const grid;

    JlCache(SpGrid const grid, int l_max);
    ~JlCache();
	
	inline
	double operator()(int ir, int il) const {
		assert(ir >= -1 && ir < grid.n[iR]);
		assert(il >= 0  && il < l_max);

		return data[ir + il*(grid.n[iR]+1) + 1];
	}

	double operator()(double r, int il) const;

	inline
	double& operator()(int ir, int il) {
		assert(ir >= -1 && ir < grid.n[iR]);
		assert(il >= 0 && il < l_max);

		return data[ir + il*(grid.n[iR]+1) + 1];
	}

    static double calc(double r, int il);
};


/*!
 * \brief [Spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics)\f$Y_l^m(\theta)\f$
 * \param[in] l
 * \param[in] m
 * \param[in] ic index of \f$\cos\theta\f$
 * */
struct YlmCache {
public:
    double* data;
	int size;
	int l_max;
	int m_max;
	SpGrid const grid;

    YlmCache(SpGrid const grid, int l_max, int m_max=2);
    ~YlmCache();
	
	inline
	double operator()(int l, int m, int ic) const {
		assert(ic >= 0 && ic < grid.n[iC]);
		assert(l >= 0 && l <= l_max);

		return data[l + ic*(l_max+1) + m*(l_max+1)*grid.n[iC]];
	}
    double operator()(int l, int m, double theta) const;

	inline
	double& operator()(int l, int m, int ic) {
		assert(ic >= 0 && ic < grid.n[iC]);
		assert(l >= 0 && l <= l_max);

		return data[l + ic*(l_max+1) + m*(l_max+1)*grid.n[iC]];
	}

    static double calc(int l, int m, double theta);
};


/*! \file
 * Свойства сферических функций
 */

/*!  \brief \f$(-1)^{p}\f$
 * */
int pow_minus_one(int p);

/*!
 * \brief [Clebsch-Gordan coefficients](https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients) \f$<j_1 m_1, j_2 m_2|JM>\f$
 *
 * Relaton to [Wigner 3-j symbols](https://en.wikipedia.org/wiki/3-j_symbol)
 * \f[<j_1m_1, j_2m_2|JM> = (-1)^{j_1-j_2+M} * \sqrt{2J + 1}(j_1 j_2 J, m_1 m_2 -M)\f]
 */
double clebsch_gordan_coef(int j1, int m1, int j2, int m2, int J, int M);

/*!
 * \f[\int_{4\pi} Y_{l_1}^{m_1*} Y_{l_2}^{m_2*} Y_L^M d\Omega =
 * \sqrt{\frac{(2l_1 + 1)(2l_2 + 1)}{4\pi(2L + 1)}}<l_10, l_20|L0><l_1m_1,l_2m_2|LM>\f]
 * */
double y3(int l1, int m1, int l2, int m2, int L, int M);

/*!
 * \brief Разложение функции по сферическим гармоникам
 * */

#ifndef __CUDACC__
#include "integrate.h"
template <typename T>
T sh_series_r(std::function<T(int, int)> f, int ir, int l, int m, SpGrid2d const& grid, YlmCache const* ylm_cache) {
	return integrate_1d_cpp<T>([f, ir, ylm_cache, l, m, grid](int ic) -> T {
            return f(ir, ic)*(*ylm_cache)(l, m, ic)*std::sin(grid.theta(ic));
			}, grid.n[iC], grid.d[iC])*2*M_PI;
}

template <typename T>
void sh_series(
		std::function<T(int, int)> func,
		int l, int m, SpGrid2d const& grid,
		T* series,
		YlmCache const* ylm_cache,
		std::optional<Range> rRange = std::nullopt) {
	auto range = rRange.value_or(grid.getFullRange(iR));

#pragma omp parallel for
	for (int ir = range.start; ir < range.end; ++ir) {
		series[ir] = sh_series_r(func, ir, l, m, grid, ylm_cache);
	}
}

#include "array.h"
template <typename T>
inline void sh_series(
		ArraySp2D<T> const* arr,
		int l, int m,
		T* series,
		YlmCache const* ylm_cache,
		std::optional<Range> rRange = std::nullopt) {
	sh_series(std::function<T(int, int)>(arr[0]), l, m, arr->grid, series, ylm_cache, rRange);
}

template <typename T>
void sh_to_sp(ArraySh<T> const* src, ArraySp2D<T>* dest, YlmCache const* ylm_cache, int m) {
	for (int ir = 0; ir < src->grid.n[iR]; ++ir) {
		for (int ic = 0; ic < dest->grid.n[iC]; ++ic) {
			T res = 0.0;
			for (int l = 0; l < src->grid.n[iL]; ++l) {
				res += (*src)(ir, l)*(*ylm_cache)(l, m, ic);
			}
			(*dest)(ir, ic) = res;
		}
	}
}

template <typename T>
void sp_to_sh(ArraySp2D<T> const* src, ArraySh<T>* dest, YlmCache const* ylm_cache, int m) {
	for (int l = 0; l < dest->grid.n[iL]; ++l) {
		sh_series<T>(src, l, m, &(*dest)(0, l), ylm_cache);
	}
}
#endif
