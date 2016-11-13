#include <complex.h>

typedef double complex cdouble;


/*
 * a[i]*y[i+1] + b[i]*y[i] + c[i]*y[i-1] = f[i]
 * (y[0] - y[-1]) + k[0]*y[1] = 0
 * (y[N] - y[N-1]) + k[1]*y[N] = 0
 *
 */
void prog(cdouble* y, cdouble const* a, cdouble const* b, cdouble const* c, cdouble const* f, double const k[2], int const N) {
	// y[i] = alpha[i]*y[i+1] + betta[i]
	cdouble* alpha = (cdouble*) malloc(sizeof(cdouble)*N);
	cdouble* betta = (cdouble*) malloc(sizeof(cdouble)*N);

	alpha[0] = -a[0] / (b[0] + c[0]*(k[0] + 1));
	betta[0] = f[0]  / (b[0] + c[0]*(k[0] + 1));
	for (int i = 1; i < N; ++i) {
		alpha[i] =                   - a[i] / (b[i] + c[i]*alpha[i-1]);
		betta[i] = (f[i] - c[i]*betta[i-1]) / (b[i] + c[i]*alpha[i-1]);
	}

	y[N-1] = betta[N-1]/(1 + (k[1] - 1)*alpha[N-1]);
	for (int i = N-2; i >= 0; --i) {
		y[i] = alpha[i]*y[i+1] + betta[i];
	}

	free(alpha);
	free(betta);
}
