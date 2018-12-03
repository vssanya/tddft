#pragma once

#include <math.h>

/* 
 *
 * Ylm - сферические функции
 * Y10 = cosθ
 * Y20 = (3cosθ^2 - 1)/2
 * 
 * clm = <Ylm|Y10|Yl+1m>
 * qlm = <Ylm|Y20|Yl+2m>
 * plm = <Ylm|Y20|Ylm>
 * */
#ifdef __CUDACC__
__host__ __device__
#endif
inline double clm(int l, int m) {
	return sqrt((double)((l+1)*(l+1) - m*m)/(double)((2*l + 1)*(2*l + 3)));
}

double qlm(int l, int m) __attribute__((pure));
double plm(int l, int m) __attribute__((pure));

double clamp(double x, double lower, double upper) __attribute__((pure));
double smoothstep(double x, double x0, double x1) __attribute__((pure));
double smoothpulse(double x, double dx_smooth, double dx_pulse) __attribute__((pure));

inline int div_up(int x, int y)
{
    return (x - 1) / y + 1;
}

void selectGpuDevice(int id);
