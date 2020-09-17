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


TDSFM_Base::TDSFM_Base(SpGrid const k_grid, ShGrid const r_grid, int ir, int m_max, bool own_data):
	k_grid(k_grid),
	r_grid(r_grid),
	ir(ir),
	jl_grid(),
    ylm_grid(),
	jl(nullptr),
    ylm(nullptr),
	data(nullptr),
	m_max(m_max),
	int_A(0.0),
	int_A2(0.0)
{
	if (own_data) {
		data = new cdouble[k_grid.n[iR]*k_grid.n[iT]]();
	}
}

TDSFM_Base::~TDSFM_Base() {
    if (ylm != nullptr) {
		delete ylm;
	}

    if (jl != nullptr) {
		delete jl;
	}

	if (data != nullptr) {
		delete[] data;
	}
}

void TDSFM_Base::init_cache() {
    if (jl == nullptr) {
		jl = new JlCache(jl_grid, r_grid.n[iL]+1);
	}

    if (ylm == nullptr) {
		ylm = new YlmCache(ylm_grid, r_grid.n[iL], m_max);
	}
}

TDSFM_E::TDSFM_E(SpGrid const k_grid, ShGrid const r_grid, double A_max, int ir, int m_max, bool init_cache, bool own_data):
	TDSFM_Base(k_grid, r_grid, ir, m_max, own_data)
{
    double k_max = k_grid.Rmax() + A_max;
    double r = r_grid.r(ir);

	int N[3] = {(int)(k_max*r/(k_grid.d[iR]*r_grid.d[iR]))*2, k_grid.n[1]*2, k_grid.n[2]};
    jl_grid = SpGrid(N, k_max*r);

	int N_ylm[3] = {(int)(k_max/k_grid.d[iR])*2, k_grid.n[1]*2, k_grid.n[2]};
    ylm_grid = SpGrid(N_ylm, k_max);

	if (init_cache) {
		this->init_cache();
	}
}

TDSFM_E::~TDSFM_E() {
}

TDSFM_A::TDSFM_A(SpGrid const k_grid, ShGrid const r_grid, int ir, int m_max, bool init_cache, bool own_data):
	TDSFM_Base(k_grid, r_grid, ir, m_max, own_data)
{
    double k_max = k_grid.Rmax();
    double r = r_grid.r(ir);

	int N[3] = {k_grid.n[0], k_grid.n[1], k_grid.n[2]};
    jl_grid = SpGrid(N, k_max*r);
    ylm_grid = SpGrid(N, k_max);

	if (init_cache) {
		this->init_cache();
	}
}

TDSFM_A::~TDSFM_A() {
}

void TDSFM_E::calc(field_t const* field, ShWavefunc const& wf, double t, double dt, double mask, cdouble* data) {
	if (data == nullptr) {
		data = this->data;
	}

    double r  = wf.grid.r(ir);
	double dr = wf.grid.d[iR];

	double At_dt = field_A(field, t-dt);
	double At    = field_A(field, t);

	int_A  += (At + At_dt)*dt*0.5;
	int_A2 += (pow(At, 2) + pow(At_dt, 2))*dt*0.5;

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<k_grid.n[iR]; ik++) {
		for (int it=0; it<k_grid.n[iT]; it++) {
            double k  = k_grid.r(ik);
            double kz = cos(k_grid.theta(it))*k;

			cdouble S = cexp(0.5*I*(k*k*t + 2*kz*int_A + int_A2));

			double k_A = sqrt(k*k + 2*kz*At + At*At);
			double k_A_z = kz + At;

			cdouble a_k = 0.0;
			for (int il=wf.m; il<wf.grid.n[iL]; il++) {
				cdouble psi = wf(ir, il);
				cdouble dpsi = (wf(ir+1, il) - wf(ir-1, il))/(2*dr);
				a_k += cpow(-I, il+1)*(
						(*jl)(k_A*r, il)*(dpsi-(il+1)*psi/r) +
						(*jl)(k_A*r, il+1)*k_A*psi
						)*(*ylm)(il, wf.m, k_A_z/k_A);
			}

			data[ik + it*k_grid.n[iR]] += a_k*r/sqrt(2.0*M_PI)*S*dt*mask;
		}
	}
}

void TDSFM_A::calc(field_t const* field, ShWavefunc const& wf, double t, double dt, double mask, cdouble* data) {
	if (data == nullptr) {
		data = this->data;
	}

    double r  = wf.grid.r(ir);
	double dr = wf.grid.d[iR];

	double At_dt = field_A(field, t-dt);
	double At    = field_A(field, t);

	int_A  += (At + At_dt)*dt*0.5;

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<k_grid.n[iR]; ik++) {
		for (int it=0; it<k_grid.n[iT]; it++) {
            double k  = k_grid.r(ik);
            double kz = k*cos(k_grid.theta(it));

			cdouble S = cexp(0.5*I*(k*k*t + 2*kz*int_A));

			cdouble a_k = 0.0;
			{
				int il = wf.m;
				cdouble psi = wf(ir, il);
				cdouble dpsi = wf.d_dr(ir, il);
				a_k += cpow(-I, il+1)*(
						(*jl)(ik, il)*(dpsi-(il+1)*psi/r) +
						(*jl)(ik, il+1)*k*psi +
						2*I*(*jl)(ik, il)*At*clm(il, wf.m)*wf(ir, il+1)
						)*(*ylm)(il, wf.m, it);
			}
			for (int il=wf.m+1; il<wf.grid.n[iL]-1; il++) {
				cdouble psi = wf(ir, il);
				cdouble dpsi = wf.d_dr(ir, il);
				a_k += cpow(-I, il+1)*(
						(*jl)(ik, il)*(dpsi-(il+1)*psi/r) +
						(*jl)(ik, il+1)*k*psi +
						2*I*(*jl)(ik, il)*At*(clm(il-1, wf.m)*wf(ir, il-1) + clm(il, wf.m)*wf(ir, il+1))
						)*(*ylm)(il, wf.m, it);
			}
			{
				int il=wf.grid.n[iL]-1;
				cdouble psi = wf(ir, il);
				cdouble dpsi = wf.d_dr(ir, il);
				a_k += cpow(-I, il+1)*(
						(*jl)(ik, il)*(dpsi-(il+1)*psi/r) +
						(*jl)(ik, il+1)*k*psi +
						2*I*(*jl)(ik, il)*At*clm(il-1, wf.m)*wf(ir, il-1)
						)*(*ylm)(il, wf.m, it);
			}

			data[ik + it*k_grid.n[iR]] += a_k*r/sqrt(2.0*M_PI)*S*dt*mask;
		}
	}
}

void TDSFM_E::calc_inner(field_t const* field, ShWavefunc const& wf, double t, int ir_min, int ir_max, int l_min, int l_max, cdouble* data) {
	if (data == nullptr) {
		data = this->data;
	}

    if (l_max == -1 || l_max > wf.grid.n[iL]) {
        l_max = wf.grid.n[iL];
    }

    double At = field_A(field, t);

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<k_grid.n[iR]; ik++) {
		for (int it=0; it<k_grid.n[iT]; it++) {
            double k  = k_grid.r(ik);
            double kz = cos(k_grid.theta(it))*k;

			cdouble S = cexp(0.5*I*(k*k*t + 2*kz*int_A + int_A2));

			double k_A = sqrt(k*k + 2*kz*At + At*At);
			double k_A_z = kz + At;

			cdouble a_k = 0.0;
			for (int il=l_min; il<l_max; il++) {
				cdouble a_kl = 0.0;
				for (int ir=ir_min; ir<ir_max; ir++) {
                    double r = wf.grid.r(ir);
					a_kl += r*wf(ir, il)*(*jl)(k_A*r, il);
				}
				a_k += a_kl*cpow(-I, il)*(*ylm)(il, wf.m, k_A_z/k_A);
			}

			data[ik + it*k_grid.n[iR]] += a_k*sqrt(2.0/M_PI)*wf.grid.d[iR]*S;
		}
	}
}

void TDSFM_A::calc_inner(field_t const* field, ShWavefunc const& wf, double t, int ir_min, int ir_max, int l_min, int l_max, cdouble* data) {
	if (data == nullptr) {
		data = this->data;
	}

    if (l_max == -1 || l_max > wf.grid.n[iL]) {
        l_max = wf.grid.n[iL];
    }

#pragma omp parallel for collapse(2)
	for (int ik=0; ik<k_grid.n[iR]; ik++) {
		for (int it=0; it<k_grid.n[iT]; it++) {
            double k  = k_grid.r(ik);
            double kz = cos(k_grid.theta(it))*k;

			cdouble S = cexp(0.5*I*(k*k*t + 2*kz*int_A));

			cdouble a_k = 0.0;
            for (int il=l_min; il<l_max; il++) {
				cdouble a_kl = 0.0;
				for (int ir=ir_min; ir<ir_max; ir++) {
                    double r = wf.grid.r(ir);
                    a_kl += r*wf(ir, il)*JlCache::calc(r*k, il);
				}
                a_k += a_kl*cpow(-I, il)*(*ylm)(il, wf.m, it);
            }

			data[ik + it*k_grid.n[iR]] += a_k*sqrt(2.0/M_PI)*wf.grid.d[iR]*S;
		}
	}
}

void TDSFM_Base::calc_norm_k(ShWavefunc const& wf, int ir_min, int ir_max, int l_min, int l_max, cdouble* data) {
	if (data == nullptr) {
		data = this->data;
	}

    if (l_max == -1 || l_max > wf.grid.n[iL]) {
        l_max = wf.grid.n[iL];
    }

#pragma omp parallel for
	for (int ik=0; ik<k_grid.n[iR]; ik++) {
		double k  = k_grid.r(ik);

		double a_k = 0.0;
		for (int il=l_min; il<l_max; il++) {
			cdouble a_kl = 0.0;
			for (int ir=ir_min; ir<ir_max; ir++) {
				double r = wf.grid.r(ir);
				a_kl += r*wf(ir, il)*JlCache::calc(r*k, il);
			}
			a_k += pow(creal(a_kl), 2) + pow(cimag(a_kl), 2);
		}

		data[ik + 0*k_grid.n[iR]] = a_k*(2.0/M_PI)*pow(wf.grid.d[iR], 2);
	}
}

double TDSFM_Base::pz() const {
	double pz = 0.0;

#pragma omp parallel for reduction(+:pz) collapse(2)
	for (int ik=0; ik<k_grid.n[iR]; ik++) {
		for (int it=0; it<k_grid.n[iT]; it++) {
            double k  = k_grid.r(ik);
            double kz = cos(k_grid.theta(it))*k;

            pz += kz*(pow(creal((*this)(ik, it)), 2) + pow(cimag((*this)(ik, it)), 2))*k*k*sin(k_grid.theta(it));
		}
	}

    return pz*k_grid.d[iR]*k_grid.d[iT]*2*M_PI;
}

double TDSFM_Base::norm() const {
	double norm = 0.0;

#pragma omp parallel for reduction(+:norm) collapse(2)
	for (int ik=0; ik<k_grid.n[iR]; ik++) {
		for (int it=0; it<k_grid.n[iT]; it++) {
            double k  = k_grid.r(ik);
            norm += (pow(creal((*this)(ik, it)), 2) + pow(cimag((*this)(ik, it)), 2))*k*k*sin(k_grid.theta(it));
		}
	}

    return norm*k_grid.d[iR]*k_grid.d[iT]*2*M_PI;
}

TDSFMOrbs::TDSFMOrbs(Orbitals<ShGrid> const& orbs, SpGrid const k_grid, int ir, workspace::Gauge gauge, bool init_cache, double A_max) {
	int m_max = orbs.atom.getMmax() + 1;

	switch (gauge) {
		case workspace::Gauge::LENGTH:
			tdsfmWf = new TDSFM_E(k_grid, orbs.grid, A_max, ir, m_max, init_cache, false);
			break;
		case workspace::Gauge::VELOCITY:
			tdsfmWf = new TDSFM_A(k_grid, orbs.grid, ir, m_max, init_cache, false);
			break;
	}

	pOrbs = new Orbitals<SpGrid2d>(orbs.atom, k_grid.getGrid2d(), orbs.mpi_comm, &orbs.ne_rank[0]);
}

TDSFMOrbs::~TDSFMOrbs() {
	delete tdsfmWf;
	delete pOrbs;
}

void TDSFMOrbs::collect(cdouble* dest) const {
	pOrbs->collect(dest);
}

void TDSFMOrbs::calc(field_t const* field, Orbitals<ShGrid> const& orbs, double t, double dt, double mask) {
	for (int ie = 0; ie < orbs.atom.countOrbs; ++ie) {
		if (orbs.wf[ie] != nullptr) {
			tdsfmWf->calc(field, *orbs.wf[ie], t, dt, mask, pOrbs->wf[ie]->data);
		}
	}
}

void TDSFMOrbs::calc_inner(field_t const* field, Orbitals<ShGrid> const& orbs, double t, int ir_min, int ir_max, int l_min, int l_max) {
	for (int ie = 0; ie < orbs.atom.countOrbs; ++ie) {
		if (orbs.wf[ie] != nullptr) {
			tdsfmWf->calc_inner(field, *orbs.wf[ie], t, ir_min, ir_max, l_min, l_max, pOrbs->wf[ie]->data);
		}
	}
}
