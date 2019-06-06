#include <cmath>
using namespace std;

long int FACTORIAL_LOOKUP_TABLE[] = {
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000
};

long int factorial(int n) {
	return FACTORIAL_LOOKUP_TABLE[n];
}

double int_func_0(double t, double x, int m) {
    return exp(-t*x*x)*pow(t, m);
}

double int_dfunc_0(double t, double x, int m) {
	if (m==0) {
		return -x*x*exp(-t*x*x);
	} else {
		return (-x*x*exp(-t*x*x)*t + m*exp(-t*x*x))*pow(t, m-1);
	}
}

double int_func(double t, double x, int m) {
    return int_func_0(t, x, m) / sqrt(1 - t);
}

double int_func_res(double x, double m) {
	double res = 0.0;

	double dt = std::min(1e-2, 1e-1 / (x*x));
	int nt = (int) (1.0 / dt);
	dt = 1.0 / nt; 

	res += 0.5*(int_func(0, x, m) + int_func(1-dt, x, m));

	for (double i=1; i<nt-1; i++) {
		res += int_func(i*dt, x, m);
	}

	res *= dt;
	res += (2*int_func_0(1, x, m) - 2.0/3.0*int_dfunc_0(1, x, m)*dt)*sqrt(dt);

	return res;
}

double w_m(double x, double m) {
	double res = 0.0;

	double dt = std::min(1e-2, 1e-1 / (x*x));
	int nt = (int) (1.0 / dt);
	dt = 1.0 / nt; 

	res += 0.5*(int_func(0, x, m) + int_func(1-dt, x, m));

	for (double i=1; i<nt-1; i++) {
		res += int_func(i*dt, x, m);
	}

	res *= dt;
	res += (2*int_func_0(1, x, m) - 2.0/3.0*int_dfunc_0(1, x, m)*dt)*sqrt(dt);

    return 0.5*pow(x,(2*m + 1))*res; //si.quad(LowLevelCallable(int_func.ctypes, ptr), 0, 1)[0]
}

    

double alpha(double gamma) {
    return 2*(asinh(gamma) - gamma/sqrt(1 + gamma*gamma));
}

double betta(double gamma) {
    return 2*gamma/sqrt(1 + gamma*gamma);
}

double Am(double freq, double gamma, double Ip, int m) {
    double res = 0.0;
    
    double nu = Ip/freq*(1.0 + 1.0/(2*gamma*gamma));
    int k0 = (int) ceil(nu);
    
#pragma omp parallel for reduction(+:res)
	for (int k=k0; k<k0+500; k++) {
		res += exp(-alpha(gamma)*(k - nu))*w_m(sqrt(betta(gamma)*(k - nu)), m);
	}
    
    return res*(4.0/sqrt(3*M_PI))/factorial(m)*(gamma*gamma/(1.0 + gamma*gamma));
}

double w_ppt(int l, int m, double Cnl, double Ip, int Z, double E, double freq) {
    if (E < 0.000054) {
        return 0.0;
	}
    
    double k = sqrt(2*Ip);
    
    double gamma = k*freq/E;
    
    double g = 3.0/(2*gamma)*((1 + 1.0/(2*gamma*gamma))*asinh(gamma) - sqrt(1 + gamma*gamma)/(2*gamma));
    
    double F = E/pow(k,3); // F/F0
    
    double res = Ip * sqrt(3.0/(2*M_PI)) * Cnl*Cnl * (2*l + 1) * factorial(l + m) / (pow(2,m) * factorial(m)*factorial(l - m));
    res *= pow((2.0/(F*sqrt(1+gamma*gamma))), (-m - 1.5 + 2*Z/k));
    res *= exp(-2.0/(3*F)*g)*Am(freq, gamma, Ip, m);
    
    return 2*res;
}

double w_ppt_Qc(int l, int m, double Cnl, double Ip, int Z, double E, double freq) {
    if (E < 0.000054) {
        return 0.0;
	}
    
    double k = sqrt(2*Ip);
    double gamma = k*freq/E;
    
    return w_ppt(l, m, Cnl, Ip, Z, E, freq)*pow(1.0+2.0*gamma/M_E,-2*Z/k);
}

double w_adk(int l, int m, double Cnl, double Ip, int Z, double E, double freq) {
    if (E < 0.000054) {
        return 0.0;
	}
    
    double k = sqrt(2*Ip);
    double F = E/pow(k,3); // F/F0
    
    double res = Ip * sqrt(3.0 * F / M_PI) * Cnl*Cnl * (2*l + 1) * factorial(l + m) / (pow(2,m) * factorial(m)*factorial(l - m));
    res *= pow(2.0/F, (-m - 1 + 2*Z/k));
    res *= exp(-2.0/(3*F));
    
    return res;
}

double w_tl_exp(double Ip, int Z, double E, double alpha) {
    double k = sqrt(2*Ip);
    double F = E/pow(k,3); // F/F0

	return exp(-alpha*(Z*Z/Ip)*F);
}
