#include "linalg.h"
#include <math.h>


void linalg::tdm_t::inv(double* b) const {
	double* theta = new double[N+1];
	double* phi = new double[N+1];

	theta[0] = 1.0;
	theta[1] = a00;
	for (int i = 2; i < N+1; ++i) {
		theta[i] = a[1]*theta[i-1] - a[0]*a[2]*theta[i-2];
	}

	phi[N]   = 1.0;
	phi[N-1] = a[1];
	for (int i = N-2; i > 0; --i) {
		phi[i] = a[1]*phi[i+1] - a[0]*a[2]*phi[i+2];
	} {
		int i = 0;
		phi[i] = a00*phi[i+1] - a[0]*a[2]*phi[i+2];
	}

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < i; ++j) {
			b[j + i*N] = pow(-1, i+j)*theta[j]*phi[i+1]*pow(a[2], i-j)/theta[N];
		}
		for (int j = i; j < N; ++j) {
			b[j + i*N] = pow(-1, i+j)*theta[i]*phi[j+1]*pow(a[0], j-i)/theta[N];
		}
	}

	delete[] theta;
	delete[] phi;
}

void linalg::tdm_t::matrix_dot(double* b) const {
	double* row = new double[N];

	for (int i=0; i<N; ++i) {
		for (int j=0; j<N; ++j) {
			row[j] = b[j + i*N];
		} 

		{ int j=0;
			b[j + i*N] =                 row[j]*a00  + row[j+1]*a[0];
		} for (int j=1; j<N; ++j) {
			b[j + i*N] = row[j-1]*a[2] + row[j]*a[1] + row[j+1]*a[0];
		}
	}

	delete[] row;
}

void linalg::eq_solve(cdouble* vec, tdm_t const& M, tdm_t const& d, cdouble* alpha, cdouble* betta) {
	int N = M.N;

	cdouble al[3];
	cdouble ar[3];
	cdouble f;

	{
		int i = 0;

		al[1] = M.a00 - d.a00;
		ar[1] = M.a00 + d.a00;
		al[2] = M.a[2] - d.a[2];
		ar[2] = M.a[2] + d.a[2];

		f = ar[1]*vec[i] + ar[2]*vec[i+1];

		alpha[0] = -al[2]/al[1];
		betta[0] = f/al[1];
	}

	for (int i = 1; i < N - 1; ++i) {
		for (int j = 0; j < 3; ++j) {
			al[j] = M.a[j] - d.a[j];
			ar[j] = M.a[j] + d.a[j];
		}

		cdouble c = al[1] + al[0]*alpha[i-1];
		f = ar[0]*vec[i-1] + ar[1]*vec[i] + ar[2]*vec[i+1];

		alpha[i] = - al[2] / c;
		betta[i] = (f - al[0]*betta[i-1]) / c;
	}

	{
		int i = N - 1;

		al[0] = M.a[0] - d.a[0];
		ar[0] = M.a[0] + d.a[0];
		al[1] = M.aNN - d.aNN;
		ar[1] = M.aNN + d.aNN;

		cdouble c = al[1] + al[0]*alpha[i-1];
		f = ar[0]*vec[i-1] + ar[1]*vec[i];

		betta[i] = (f - al[0]*betta[i-1]) / c;
	}

	vec[N-1] = betta[N-1];
	for (int i = N-2; i >= 0; --i) {
		vec[i] = alpha[i]*vec[i+1] + betta[i];
	}
}
