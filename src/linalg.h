#pragma once

#include "types.h"

#include <functional>


namespace linalg
{
	struct tdm_t {
		double a00; // top-left element
		double aNN; // bottom-right element
		double a[3];
		int N;

		/*!
		 * \brief Inverse of a tridiagonal matrix (a)
		 * */
		void inv(double* b) const;

		/*!
		 * \brief Multiply matrix (a) on tridiagonal matrix (b)
		 * */
		void matrix_dot(double* b) const;
	};

	// Solve (M-d)x = (M+d) vec
	// Result x write to vec
	void eq_solve(
			cdouble* vec,
			tdm_t const& M,
			tdm_t const& d,
			cdouble* alpha,
			cdouble* betta
			);

	// matrix = [[0, 1], [ 1, 0]]
	namespace matrix_bE
	{
		const cdouble eigenval[2] = {-1.0, 1.0};

		inline void dot(cdouble v[2]) {
			cdouble res[2] = {
				v[0] + v[1],
				-v[0] + v[1]
			};

			v[0] = res[0];
			v[1] = res[1];
		}

		inline void dot_T(cdouble v[2]) {
			cdouble res[2] = {
				0.5*(v[0] - v[1]),
				0.5*(v[0] + v[1])
			};

			v[0] = res[0];
			v[1] = res[1];
		}

#ifdef __CUDACC__
		__device__ void dot(cuComplex v[2]) {
			cuComplex res[2] = {
				v[0] + v[1],
				-v[0] + v[1]
			};

			v[0] = res[0];
			v[1] = res[1];
		}

		__device__ void dot_T(cuComplex v[2]) {
			cuComplex res[2] = {
				0.5*(v[0] - v[1]),
				0.5*(v[0] + v[1])
			};

			v[0] = res[0];
			v[1] = res[1];
		}
#endif
	} /* matrix_bE */ 

	// matrix = [[0, 1], [-1, 0]]
	namespace matrix_bA
	{
		const cdouble eigenval[2] = {-I, I};

		inline void dot(cdouble v[2]) {
			cdouble res[2] = {
				-I*v[0] + v[1],
				I*v[0] + v[1]
			};

			v[0] = res[0];
			v[1] = res[1];
		}

		inline void dot_T(cdouble v[2]) {
			cdouble res[2] = {
				0.5*(I*v[0] - I*v[1]),
				0.5*(v[0] + v[1])
			};

			v[0] = res[0];
			v[1] = res[1];
		}
	} /* matrix_bA */ 

	typedef std::function<void(cdouble[2])> matrix_f;

	inline void matrix_dot_vec(int N, cdouble* v[2], matrix_f dot) {
		cdouble x[2];
#pragma omp for private(x)
		for (int i=0; i<N; i++) {
			x[0] = v[0][i];
			x[1] = v[1][i];

			dot(x);

			v[0][i] = x[0];
			v[1][i] = x[1];
		}
	}
} /* linalg */ 
