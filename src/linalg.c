#include "linalg.h"

#include "stdlib.h"
#include "math.h"


void linalg_tdm_inv(double a00, double const a[3], int N, double b[N*N]) {
	double* theta = malloc(sizeof(double)*(N+1));
	double* phi = malloc(sizeof(double)*(N+1));

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

	free(theta);
	free(phi);
}

void linalg_m_dot_tdm(int N, double a[N*N], double b00, double const b[3]) {
	double* row = malloc(sizeof(double)*N);
	for (int i=0; i<N; ++i) {
		for (int j=0; j<N; ++j) {
			row[j] = a[j + i*N];
		}

		{ int j=0;
			a[j + i*N] =                 row[j]*b00  + row[j+1]*b[0];
		} for (int j=1; j<N; ++j) {
			a[j + i*N] = row[j-1]*b[2] + row[j]*b[1] + row[j+1]*b[0];
		}
	}
	free(row);
}
