#include "utils.h"

#include <math.h>

// clm = <Yl,m|cosθ|Yl+1,m>
// Yl,m - сферические функции
double clm(int l, int m) {
	return sqrt((double)((l+1)*(l+1) - m*m)/(double)((2*l + 1)*(2*l + 3)));
}

double min(double a, double b) {
	return a < b ? a : b;
}

double max(double a, double b) {
	return a > b ? a : b;
}

double clamp(double x, double lower, double upper) {
	return min(upper, max(x, lower));
}

double smoothstep(double x, double x0, double x1) {
	double t = clamp((x-x0)/(x1-x0), 0.0, 1.0);
	return pow(t,3)*(t*(t*6.0 - 15.0) + 10.0);
}

double smoothpulse(double x, double dx_smooth, double dx_pulse) {
	double x_center = dx_smooth + 0.5*dx_pulse;
	x = x < x_center ? x : 2*x_center - x;
	return smoothstep(x, 0.0, dx_smooth);
}
