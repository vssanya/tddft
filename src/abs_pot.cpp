#include "abs_pot.h"

#include "utils.h"
#include <math.h>


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

UabsMultiHump::UabsMultiHump(double l_min, double l_max) {
    static func_t const f_e_opt[3] = {uabs_func_cos_e_opt, uabs_func_pt_e_opt, uabs_func_pt_e_opt};
    static int const N = 2;

    for (int i = 0; i < N; ++i) {
        l[i] = l_max*pow(l_min/l_max, i/(double)(N-1));
        a[i] = f_e_opt[i](2*M_PI) / (l[i]*l[i]);
    }
}

double UabsMultiHump::u(const ShGrid& grid, double r) const {
	static func_t const f[3] = {uabs_func_cos, uabs_func_pt, uabs_func_pt};

    double const r_max = grid.Rmax();

	double result = 0.0;
    for (int i = 0; i < 2; ++i) {
        result += a[i]*f[i]((r - r_max + l[i]) / l[i]);
	}

	return result;
}

UabsCache::UabsCache(const Uabs &uabs, const ShGrid &grid, double *u):
    uabs(uabs),
    grid(grid) {
    data = new double[grid.n[iR]];

    if (u != nullptr) {
        for (int ir=0; ir<grid.n[iR]; ir++) {
            data[ir] = u[ir];
        }
    } else {
        for (int ir=0; ir<grid.n[iR]; ir++) {
            data[ir] = uabs.u(grid, grid.r(ir));
        }
    }
}

UabsCache::~UabsCache() {
    delete[] data;
}

double uabs(ShGrid const* grid, int ir, int il, int im) {
    double const r = grid->r(ir);
	double r_max = grid->Rmax();
	double dr = 0.2*r_max;
	return 10*smoothstep(r, r_max-dr, r_max);
}

double mask_core(ShGrid const* grid, int ir, int il, int im) {
	double const r = grid->r(ir);
	double const r_core = 10.0;
	double const dr = 2.0;
	return smoothstep(r_core + dr - r, 0.0, dr);  
}

