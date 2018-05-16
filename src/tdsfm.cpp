/*
 * =====================================================================================
 *
 *       Filename:  tdsfm.c
 *
 *    Description:  Time-dependent surface flux method
 *
 *        Version:  1.0
 *        Created:  11.10.2017 14:59:40
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Romanov Alexander, 
 *   Organization:  IAP RAS
 *
 * =====================================================================================
 */

#include <stdlib.h>
#include <stdio.h>

#include "tdsfm.h"
#include "integrate.h"
#include "utils.h"


TDSFM_Base::TDSFM_Base(SpGrid const* k_grid, ShGrid const* r_grid, int ir):
	k_grid(k_grid),
	r_grid(r_grid),
	ir(ir),
	jl_grid(nullptr),
    ylm_grid(nullptr),
	jl(nullptr),
    ylm(nullptr),
	int_A(0.0),
	int_A2(0.0)
{
	data = new cdouble[k_grid->n[iR]*k_grid->n[iC]]();
}

TDSFM_Base::~TDSFM_Base() {
    if (ylm != nullptr) {
		delete ylm;
	}

    if (jl != nullptr) {
		delete jl;
	}

	delete[] data;
}

void TDSFM_Base::init_cache() {
    if (jl == nullptr) {
		jl = new JlCache(jl_grid, r_grid->n[iL]+1);
	}

    if (ylm == nullptr) {
		ylm = new YlmCache(ylm_grid, r_grid->n[iL]);
	}
}

TDSFM_E::TDSFM_E(SpGrid const* k_grid, ShGrid const* r_grid, double A_max, int ir, bool init_cache):
	TDSFM_Base(k_grid, r_grid, ir)
{
    double k_max = k_grid->Rmax() + A_max;
    double r = r_grid->r(ir);

	int N[3] = {(int)(k_max*r/(k_grid->d[iR]*r_grid->d[iR]))*2, k_grid->n[1]*2, k_grid->n[2]};
    jl_grid = new SpGrid(N, k_max*r);

	int N_ylm[3] = {(int)(k_max/k_grid->d[iR])*2, k_grid->n[1]*2, k_grid->n[2]};
    ylm_grid = new SpGrid(N_ylm, k_max);

	if (init_cache) {
		this->init_cache();
	}
}

TDSFM_E::~TDSFM_E() {
    delete jl_grid;
    delete ylm_grid;
}

TDSFM_A::TDSFM_A(SpGrid const* k_grid, ShGrid const* r_grid, int ir, bool init_cache):
	TDSFM_Base(k_grid, r_grid, ir)
{
    double k_max = k_grid->Rmax();
    double r = r_grid->r(ir);

	int N[3] = {k_grid->n[0], k_grid->n[1], k_grid->n[2]};
    jl_grid = new SpGrid(N, k_max*r);
    ylm_grid = new SpGrid(N, k_max);

	if (init_cache) {
		this->init_cache();
	}
}

TDSFM_A::~TDSFM_A() {
    delete jl_grid;
    delete ylm_grid;
}

void TDSFM_E::calc(field_t const* field, ShWavefunc const& wf, double t, double dt, double mask) {
    double r  = wf.grid->r(ir);
	double dr = wf.grid->d[iR];

	double At_dt = field_A(field, t-dt);
	double At    = field_A(field, t);

	int_A  += (At + At_dt)*dt*0.5;
	int_A2 += (pow(At, 2) + pow(At_dt, 2))*dt*0.5;

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<k_grid->n[iR]; ik++) {
		for (int ic=0; ic<k_grid->n[iC]; ic++) {
            double k  = k_grid->r(ik);
            double kz = k_grid->c(ic)*k;

			cdouble S = cexp(0.5*I*(k*k*t + 2*kz*int_A + int_A2));

			double k_A = sqrt(k*k + 2*kz*At + At*At);
			double k_A_z = kz + At;

			cdouble a_k = 0.0;
			for (int il=0; il<wf.grid->n[iL]; il++) {
				cdouble psi = wf(ir, il);
				cdouble dpsi = (wf(ir+1, il) - wf(ir-1, il))/(2*dr);
				a_k += cpow(-I, il+1)*(
						(*jl)(k_A*r, il)*(dpsi-(il+1)*psi/r) +
						(*jl)(k_A*r, il+1)*k_A*psi
						)*(*ylm)(il, wf.m, k_A_z/k_A);
			}

            (*this)(ik, ic) += a_k*r/sqrt(2.0*M_PI)*S*dt*mask;
		}
	}
}

void TDSFM_A::calc(field_t const* field, ShWavefunc const& wf, double t, double dt, double mask) {
    double r  = wf.grid->r(ir);
	double dr = wf.grid->d[iR];

	double At_dt = field_A(field, t-dt);
	double At    = field_A(field, t);

	int_A  += (At + At_dt)*dt*0.5;

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<k_grid->n[iR]; ik++) {
		for (int ic=0; ic<k_grid->n[iC]; ic++) {
            double k  = k_grid->r(ik);
            double kz = k_grid->c(ic)*k;

			cdouble S = cexp(0.5*I*(k*k*t + 2*kz*int_A));

			cdouble a_k = 0.0;
			{
				int il = 0;
				cdouble psi = wf(ir, il);
				cdouble dpsi = wf.d_dr(ir, il);
				a_k += cpow(-I, il+1)*(
						(*jl)(ik, il)*(dpsi-(il+1)*psi/r) +
						(*jl)(ik, il+1)*k*psi +
						2*I*(*jl)(ik, il)*At*clm(il, wf.m)*wf(ir, il+1)
						)*(*ylm)(il, wf.m, ic);
			}
			for (int il=1; il<wf.grid->n[iL]-1; il++) {
				cdouble psi = wf(ir, il);
				cdouble dpsi = wf.d_dr(ir, il);
				a_k += cpow(-I, il+1)*(
						(*jl)(ik, il)*(dpsi-(il+1)*psi/r) +
						(*jl)(ik, il+1)*k*psi +
						2*I*(*jl)(ik, il)*At*(clm(il-1, wf.m)*wf(ir, il-1) + clm(il, wf.m)*wf(ir, il+1))
						)*(*ylm)(il, wf.m, ic);
			}
			{
				int il=wf.grid->n[iL]-1;
				cdouble psi = wf(ir, il);
				cdouble dpsi = wf.d_dr(ir, il);
				a_k += cpow(-I, il+1)*(
						(*jl)(ik, il)*(dpsi-(il+1)*psi/r) +
						(*jl)(ik, il+1)*k*psi +
						2*I*(*jl)(ik, il)*At*clm(il-1, wf.m)*wf(ir, il-1)
						)*(*ylm)(il, wf.m, ic);
			}

            (*this)(ik, ic) += a_k*r/sqrt(2.0*M_PI)*S*dt*mask;
		}
	}
}

void TDSFM_E::calc_inner(field_t const* field, ShWavefunc const& wf, double t, int ir_min, int ir_max, int l_max) {
    if (l_max == -1 || l_max > wf.grid->n[iL]) {
        l_max = wf.grid->n[iL];
    }

    double At = field_A(field, t);

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<k_grid->n[iR]; ik++) {
		for (int ic=0; ic<k_grid->n[iC]; ic++) {
            double k  = k_grid->r(ik);
            double kz = k_grid->c(ic)*k;

			cdouble S = cexp(0.5*I*(k*k*t + 2*kz*int_A + int_A2));

			double k_A = sqrt(k*k + 2*kz*At + At*At);
			double k_A_z = kz + At;

			cdouble a_k = 0.0;
			for (int il=0; il<wf.grid->n[iL]; il++) {
				cdouble a_kl = 0.0;
				for (int ir=ir_min; ir<ir_max; ir++) {
                    double r = wf.grid->r(ir);
					a_kl += r*wf(ir, il)*(*jl)(k_A*r, il);
				}
				a_k += a_kl*cpow(-I, il)*(*ylm)(il, wf.m, k_A_z/k_A);
			}

			(*this)(ik, ic) += a_k*sqrt(2.0/M_PI)*wf.grid->d[iR]*S;
		}
	}
}

void TDSFM_A::calc_inner(field_t const* field, ShWavefunc const& wf, double t, int ir_min, int ir_max, int l_max) {
    if (l_max == -1 || l_max > wf.grid->n[iL]) {
        l_max = wf.grid->n[iL];
    }

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<k_grid->n[iR]; ik++) {
		for (int ic=0; ic<k_grid->n[iC]; ic++) {
            double k  = k_grid->r(ik);
            double kz = k_grid->c(ic)*k;

			cdouble S = cexp(0.5*I*(k*k*t + 2*kz*int_A));

			cdouble a_k = 0.0;
            for (int il=0; il<l_max; il++) {
				cdouble a_kl = 0.0;
				for (int ir=ir_min; ir<ir_max; ir++) {
                    double r = wf.grid->r(ir);
                    a_kl += r*wf(ir, il)*JlCache::calc(r*k, il);
				}
                //a_k += a_kl*cpow(-I, il)*(*ylm)(il, wf.m, ic);
                a_k += a_kl*cpow(-I, il)*YlmCache::calc(il, wf->m, k_grid->theta(ic));
            }

			(*this)(ik, ic) += a_k*sqrt(2.0/M_PI)*wf.grid->d[iR]*S;
		}
	}
}

double TDSFM_Base::pz() const {
	double pz = 0.0;

#pragma omp parallel for reduction(+:pz) collapse(2)
	for (int ik=0; ik<k_grid->n[iR]; ik++) {
		for (int ic=0; ic<k_grid->n[iC]; ic++) {
            double k  = k_grid->r(ik);
            //double kz = k_grid->c(ic)*k;
            double kz = cos(k_grid->theta(ic))*k;

            pz += kz*(pow(creal((*this)(ik, ic)), 2) + pow(cimag((*this)(ik, ic)), 2))*k*k*sin(k_grid->theta(ic));
		}
	}

    return pz*k_grid->d[iR]*k_grid->dtheta*2*M_PI;
}
