/*!
 * /file
 * Функции численного интегрирования
 * */

#pragma once
#include <assert.h>

typedef double (*func_1d_t)(int i);
typedef double (*func_1d_data_t)(void* data, int i);
typedef double (*func_2d_t)(int ix, int iy);
typedef double (*func_3d_t)(int ix, int iy, int iz);
/*!
 * \brief Одномерное интегрирование
 * \param[in] f
 * \param[in] nx
 * \param[in] dx
 * \return \f$\int_0^X f dx = 0.5*\sum_i (f_i + f_{i+1})\f$
 * */
inline double integrate_trapezoid_1d(func_1d_t f, int nx, double dx) {
	assert(nx > 1);

	double res = 0.5*f(0);
	for (int ix = 1; ix < nx-1; ++ix) {
		res += f(ix);
	}
	res += 0.5*f(nx-1);

	return res*dx;
}

inline double integrate_trapezoid_1d_2(func_1d_t f, int ix0, int ix1, double dx) {
	assert((ix1 - ix0) > 1);

	double res = 0.5*f(ix0);
	for (int ix = ix0+1; ix < ix1-1; ++ix) {
		res += f(ix);
	}
	res += 0.5*f(ix1-1);

	return res*dx;
}


inline double integrate_trapezoid_data_1d(func_1d_data_t f, void* data, int nx, double dx) {
	assert(nx > 1);

	double res = 0.5*f(data, 0);
	for (int ix = 1; ix < nx-1; ++ix) {
		res += f(data, ix);
	}
	res += 0.5*f(data, nx-1);

	return res*dx;
}

//inline double integrate_simpsone_1d(func_1d_t f, int nx, double dx) {
inline double integrate_1d(func_1d_t f, int nx, double dx) {
	assert(nx > 1);

	double res = 0.0;
	for (int ix = 1; ix < nx-1; ix+=2) {
		res += f(ix-1) + 4*f(ix) + f(ix+1);
	}
	res *= dx/3;

	if (nx % 2 == 0) {
		res += (f(nx-2) + f(nx-1))*dx*0.5;
	}

	return res;
}

inline double integrate_data_1d(func_1d_data_t f, void* data, int nx, double dx) {
	assert(nx > 1);

	double res = 0.0;
	for (int ix = 1; ix < nx-1; ix+=2) {
		res += f(data, ix-1) + 4*f(data, ix) + f(data, ix+1);
	}
	res *= dx/3;

	if (nx % 2 == 0) {
		res += (f(data, nx-2) + f(data, nx-1))*dx*0.5;
	}

	return res;
}

inline double integrate_simpson_data_1d(func_1d_data_t f, void* data, int nx, double dx) {
	assert(nx > 1);

	double res = 0.0;
	for (int ix = 0; ix < nx-3; ix+=3) {
		res += f(data, ix) + 3*f(data, ix+1) + 3*f(data, ix+2) + f(data, ix+3);
	}
	res *= 3*dx/8;

	return res;
}

inline double integrate_bool_data_1d(func_1d_data_t f, void* data, int nx, double dx) {
	assert(nx > 1);

	double res = 0.0;
	for (int ix = 2; ix < nx-2; ix+=4) {
		res += 7*f(data, ix-2) + 32*f(data, ix-1) + 12*f(data, ix) + 32*f(data, ix+1) + 7*f(data, ix+2);
	}
	res *= 4*dx/90;

	return res;
}
