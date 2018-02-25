#include "abs_pot.h"

#include "utils.h"
#include <math.h>


double uabs_get(uabs_sh_t const* self, ShGrid const* grid, int ir, int il, int im) {
	return self->func(self, grid, ir, il, im);
}

double uabs_func_cos(double x) {
	if (fabs(x) >= 1) {
		return 0.0;
	}

	return pow(cos(M_PI*x/2), 2);
}

double uabs_func_cos_e_opt(double k) {
	return 2.15 + 0.65*k*k;
}

double uabs_func_pt(double x) {
	double const alpha = 2.0*acosh(sqrt(2.0));
	return pow(cosh(alpha*x), -2);
}

double uabs_func_pt_e_opt(double k) {
	return 3.7 + 1.14*k*k;
}

typedef double (*func_t) (double);

uabs_multi_hump_t* uabs_multi_hump_new(double lambda_min, double lambda_max) {
	static func_t const f_e_opt[3] = {uabs_func_cos_e_opt, uabs_func_pt_e_opt, uabs_func_pt_e_opt};
	static int const N = 2;

    uabs_multi_hump_t* self = (uabs_multi_hump_t*)malloc(sizeof(uabs_multi_hump_t));
    self->func = (uabs_func_t)uabs_multi_hump_func;

	for (int i = 0; i < N; ++i) {
		self->l[i] = lambda_max*pow(lambda_min/lambda_max, i/(double)(N-1));
		self->u[i] = f_e_opt[i](2*M_PI) / (self->l[i]*self->l[i]);
	}

	return self;
}

double uabs_multi_hump_func(uabs_multi_hump_t const* self, ShGrid const* grid, int ir, int il, int im) {
	static func_t const f[3] = {uabs_func_cos, uabs_func_pt, uabs_func_pt};

	double r_max = grid->Rmax();
    double const r = grid->r(ir);

	double result = 0.0;
	for (int i = 0; i < 3; ++i) {	
		result += self->u[i]*f[i]((r - r_max + self->l[i]) / self->l[i]);
	}

	return result;
}


double uabs(ShGrid const* grid, int ir, int il, int im) {
    double const r = grid->r(ir);
	double r_max = grid->Rmax();
	double dr = 0.2*r_max;
	return 10*smoothstep(r, r_max-dr, r_max);
}

double uabs_zero_func(uabs_sh_t const* uabs, ShGrid const* grid, int ir, int il, int im) {
	return 0.0;
}

double mask_core(ShGrid const* grid, int ir, int il, int im) {
	double const r = grid->r(ir);
	double const r_core = 10.0;
	double const dr = 2.0;
	return smoothstep(r_core + dr - r, 0.0, dr);  
}

