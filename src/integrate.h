/*!
 * /file
 * Функции численного интегрирования
 * */

#pragma once
#include <assert.h>

#include <functional>

template<typename T>
T integrate_1d_cpp(std::function<T(int i)> f, int nx, double dx) {
	assert(nx > 1);

	T res = 0.0;
	for (int ix = 1; ix < nx-1; ix+=2) {
		res += f(ix-1) + 4*f(ix) + f(ix+1);
	}
	res *= dx/3.0;

	if (nx % 2 == 0) {
		res += (f(nx-2) + f(nx-1))*dx*0.5;
	}

	return res;
}

template <typename T>
T integrate_1d_trap(T* f, T* dx, int Nx) {
	assert(Nx > 1);

	T res = 0.0;
	for (int ix = 0; ix < Nx-1; ++ix) {
		res += (f[ix] + f[ix+1])*dx[ix];
	}

	return res;
}

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
double integrate_trapezoid_1d(func_1d_t f, int nx, double dx);

double integrate_trapezoid_1d_2(func_1d_t f, int ix0, int ix1, double dx);
double integrate_trapezoid_data_1d(func_1d_data_t f, void* data, int nx, double dx);

// double integrate_simpsone_1d(func_1d_t f, int nx, double dx) {
double integrate_1d(func_1d_t f, int nx, double dx);
double integrate_1d_2(func_1d_t f, int ix0, int ix1, double dx);
double integrate_data_1d(func_1d_data_t f, void* data, int nx, double dx);
double integrate_simpson_data_1d(func_1d_data_t f, void* data, int nx, double dx);
double integrate_bool_data_1d(func_1d_data_t f, void* data, int nx, double dx);
