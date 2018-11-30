#include "abs_pot.h"

#include "utils.h"
#include <math.h>


UabsMultiHump::UabsMultiHump(double l_min, double l_max, std::vector<Hump> humps, double shift):
    humps(humps), l(humps.size()), a(humps.size()), shifts(humps.size()) {
    init(l_min, l_max, shift);
}

UabsMultiHump::UabsMultiHump(double l_min, double l_max, int n, double shift): humps(n), l(n), a(n), shifts(n) {
	humps[0] = CosHump;

    for (int i=1; i<n; i++) {
        humps[i] = PTHump;
    }

    init(l_min, l_max, shift);
}

void UabsMultiHump::init(double l_min, double l_max, double shift) {
    const int N = humps.size();
    for (int i = 0; i < N; ++i) {
        l[i] = l_max*pow(l_min/l_max, i/(double)(N-1));
        a[i] = humps[i].a_opt(l[i]);
    }

	shifts[N-1] = 0.0;

	for (int i = N-2; i >= 0; --i) {
		shifts[i] = shifts[i+1] + l[i+1]*shift;
	}
}

double UabsMultiHump::getHumpAmplitude(int i) const {
	return a[i];
}

void UabsMultiHump::setHumpAmplitude(int i, double value) {
	a[i] = value;
}

double UabsMultiHump::u(const ShGrid& grid, double r) const {
    double const r_max = grid.Rmax();

	double result = 0.0;
    for (int i = 0; i < humps.size(); ++i) {
        result += a[i]*humps[i].u((r - r_max + l[i] + shifts[i]) / l[i]);
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

