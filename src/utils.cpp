#include "utils.h"

#include <math.h>
#include <algorithm>

double qlm(int l, int m) {
	return 1.5/(2*l + 3)*sqrt((pow(l+1, 2) - pow(m, 2))*(pow(l+2, 2) - pow(m, 2))/(double)((2*l + 1)*(2*l + 5)));
}

double plm(int l, int m) {
	return (l*(l+1) - 3*m*m)/(double)((2*l - 1)*(2*l + 3));
}

double clamp(double x, double lower, double upper) {
    return std::min(upper, std::max(x, lower));
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
