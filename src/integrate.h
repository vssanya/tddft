/*!
 * /file
 * Функции численного интегрирования
 * */

#pragma once

typedef double (*func_1d_t)(int i);
typedef double (*func_2d_t)(int ix, int iy);
typedef double (*func_3d_t)(int ix, int iy, int iz);
/*!
 * \brief Одномерное интегрирование
 * \param[in] f
 * \param[in] nx
 * \param[in] dx
 * \return \f$\int_0^X f dx = 0.5*\sum_i (f_i + f_{i+1})\f$
 * */
inline double integrate_1d(func_1d_t f, int nx, double dx) {
	assert(nx > 1);

	double res = 0.5*f(0);
	for (int ix = 1; ix < nx-1; ++ix) {
		res += f(ix);
	}
	res += 0.5*f(nx-1);

	return res*dx;
}