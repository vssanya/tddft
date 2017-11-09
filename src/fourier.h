#pragma once

#include "types.h"


#ifdef __cplusplus
extern "C" {
#endif

void fourier_update(cdouble* f1, double const f2, double N, double dw, double t, double dt) {
	double w = 0.0;
	for (int i = 0; i < N; ++i) {
		f1[i] += cexp(I*w*t)*(dt/(2*M_PI))*f2;
		w += dw;
	}
}

#ifdef __cplusplus
}
#endif
