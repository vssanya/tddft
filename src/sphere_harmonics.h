#pragma once

#include "grid.h"


struct jl_cache_t {
	double* data;

	int l_max;
	sp_grid_t const* grid;

#ifdef __cplusplus
	jl_cache_t(sp_grid_t const* grid, int l_max);
	~jl_cache_t();
	
	inline
	double operator()(int ir, int il) const {
		return data[ir + il*grid->n[iR]];
	}
	double operator()(double r, int il) const;

	inline
	double& operator()(int ir, int il) {
		return data[ir + il*grid->n[iR]];
	}
#endif
};


struct ylm_cache_t {
	double* data;
	int size;
	int l_max;
	sp_grid_t const* grid;

#ifdef __cplusplus
	ylm_cache_t(sp_grid_t const* grid, int l_max);
	~ylm_cache_t();
	
	inline
	double operator()(int l, int m, int ic) const {
		return data[l + ic*(l_max+1) + m*(l_max+1)*grid->n[iC]];
	}
	double operator()(int l, int m, double c) const;

	inline
	double& operator()(int l, int m, int ic) {
		return data[l + ic*(l_max+1) + m*(l_max+1)*grid->n[iC]];
	}
#endif
};


#ifdef __cplusplus
extern "C" {
#endif

typedef struct jl_cache_t jl_cache_t;
typedef struct ylm_cache_t ylm_cache_t;

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
 * \brief [Spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics)\f$Y_l^m(\theta)\f$
 * \param[in] l
 * \param[in] m
 * \param[in] ic index of \f$\cos\theta\f$
 * */
ylm_cache_t* ylm_cache_new(int l_max, sp_grid_t const* grid);
void ylm_cache_del(ylm_cache_t* ylm_cache);
double ylm_cache_get(ylm_cache_t const* cache, int l, int m, int ic);
double ylm_cache_calc(ylm_cache_t const* cache, int l, int m, double c);

/*!
 * \brief Разложение функции по сферическим гармоникам
 * */
void sh_series(double (*func)(int ix, int iy), int l, int m, sp_grid_t const* grid, double* series, ylm_cache_t const* ylm_cache);

#ifdef __cplusplus
}
#endif
