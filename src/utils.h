#pragma once

// clm = <Yl,m|cosθ|Yl+1,m>
// Yl,m - сферические функции
double clm(int l, int m) __attribute__((pure));
double min(double a, double b) __attribute__((pure));
double max(double a, double b) __attribute__((pure));
double clamp(double x, double lower, double upper) __attribute__((pure));
double smoothstep(double x, double x0, double x1) __attribute__((pure));
double smoothpulse(double x, double dx_smooth, double dx_pulse) __attribute__((pure));
