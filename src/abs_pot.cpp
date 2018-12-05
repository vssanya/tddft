#include "abs_pot.h"

#include "types.h"
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

double UabsMultiHump::getWidth() const {
	return shifts[0] + 2*l[0];
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

/*
 * this program solves the Schordinger equation 
 * d^2 psi/d x^2= -(kappa^2 + 2 i epsilon v(x)) psi . It finds 
 * coefficients of transmission and reflection, T = |t|^2 and R=|r|^2, 
 * psi= t e^{- i kappa x}, x -> +infty, 
 * psi= e^{i kappa x} + r e^{-i kappa x}, x -> -infty .
 * 
  */
void Uabs::calcAbs(int N, double const* l, double* res) const {
	const double dw = 200;
	const double width = (getWidth() + dw) + 2*dw;
	int n[2] = {1, 1};
	auto grid = ShGrid(n, getWidth());
	
	auto Uabs = [this, dw, width, grid](const double x) -> double {
		if (x < dw || x > width - dw) {
			return  0.0;
		}

		return u(grid, width - dw - x);
	};

	auto runge_kutta = [Uabs](const double x, cdouble& phi, cdouble& p, const double dx, const double k) {
		cdouble g_phi[4], g_p[4];

		g_phi[0]=dx*p;
		g_p[0]=dx*2.0*(Uabs(x)*phi + I*k*p);
		g_phi[1]=dx*(p+0.5*g_p[0]);
		g_p[1]=dx*2.0*(Uabs(x+0.5*dx)*(phi+0.5*g_phi[0]) + I*k*(p+0.5*g_p[0]));
		g_phi[2]=dx*(p+0.5*g_p[1]);
		g_p[2]=dx*2.0*(Uabs(x+0.5*dx)*(phi+0.5*g_phi[1]) + I*k*(p+0.5*g_p[1]));
		g_phi[3]=dx*(p+g_p[2]);
		g_p[3]=dx*2.0*(Uabs(x+    dx)*(phi+g_phi[2])     + I*k*(p+g_p[2]));

		phi+=(g_phi[0]+2.0*g_phi[1]+2.0*g_phi[2]+g_phi[3]) / 6.0;
		p  +=(g_p[0]  +2.0*g_p[1]  +2.0*g_p[2]  +g_p[3])   / 6.0;
	};


	auto solution_equation = [width, runge_kutta](const double k) -> double {
		const double dx=1e-3;
		double x = 0.0;

		cdouble p = 0.0; 
		cdouble phi = 1.0;

		for (double x=0; x<width; x+=dx)
		{
			runge_kutta(x, phi, p, dx, k);
		}

		auto h=-p/(-p + 2.0*I*phi*k);
		auto r=-h*cexp(-I*2.0*k*x);
		auto t=(1.0-h)/phi;

		return pow(cabs(r), 2) + pow(cabs(t), 2);
	};


#pragma omp parallel for
	for (int j=0; j<N; j++) {
		auto k =2.0*M_PI/l[j]; 
		res[j] = solution_equation(k); 
	}
}
