#pragma once

#ifdef __cplusplus
extern "C" {
#endif

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
double clm(int l, int m) __attribute__((pure));
double qlm(int l, int m) __attribute__((pure));
double plm(int l, int m) __attribute__((pure));

double min(double a, double b) __attribute__((pure));
double max(double a, double b) __attribute__((pure));

double clamp(double x, double lower, double upper) __attribute__((pure));
double smoothstep(double x, double x0, double x1) __attribute__((pure));
double smoothpulse(double x, double dx_smooth, double dx_pulse) __attribute__((pure));

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#ifdef __cplusplus
}
#endif
