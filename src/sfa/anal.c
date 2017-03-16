#include "anal.h"

#include <math.h>

#include "../integrate.h"


double Fn(double n0, double g, double pn, double pz, double a, double phi) {
	return cosh(2.0*n0*a*g*cos(phi)*(2.0/3.0 + pow(pn, 2) - pow(pz, 2)))*exp(2.0*n0*g/sqrt(pow(g,2) + 1.0)*pz*(pz - a*g*sin(phi)) - 2.0*n0*pow(pn, 2)*asinh(g));
}

double Qc(double g, double w) {
	return pow(2.0*g/w / (1.0+2.0*g/M_E), 2);
}

double Ip_Up(double g, double a) {
	return 0.5 + 1.0/(4.0*pow(g,2))*(1 + pow(a,2)/4.0);
}

double C1(double w, double g, double a) {
	double g_1_2 = sqrt(1+pow(g,2));
	return 2.0*w*g*Qc(g,w)/(M_PI*g_1_2)*exp(-2.0*Ip_Up(g,a)/w*(asinh(g) - g*g_1_2/(2*pow(g,2) + 1)));
}

double pn(int n, double n0, double g) {
	return sqrt(n/n0 - (1.0 + 1.0/(2.0*pow(g,2))));
}

int n_min(double n0, double g) {
	return (int)ceil((1.0 + 1.0/(2.0*pow(g,2)))*n0);
}

double sfa_djdt(double E, double w, double a, double phi) {
	double n0 = 0.5/w;
	double g = w/E;

	double dp = 1e-3;
	double res = 0.0;

	for (int in = n_min(n0,g); in < n_min(n0,g) + 100; ++in) {
		double p = pn(in, n0, g);
		int size = (int)(2.0*p/dp);

		double func(int i) {
			double pz = - p + dp*i;
			return Fn(n0, g, p, pz, a, phi)*pz;
		}

		res += integrate_1d(func, size, dp);
	}

	return -C1(w,g,a)*res;
}

double sfa_dwdt(double E, double w, double a, double phi) {
	double n0 = 0.5/w;
	double g = w/E;

	double dp = 1e-3;
	double res = 0.0;

	for (int in = n_min(n0,g); in < n_min(n0,g) + 100; ++in) {
		double p = pn(in, n0, g);
		int size = (int)(2*p/dp);

		double func(int i) {
			double pz = - p + dp*i;
			return Fn(n0, g, p, pz, a, phi);
		}

		res += integrate_1d(func, size, dp);
	}

	return C1(w,g,a)*res;
}
