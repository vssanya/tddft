#include "sh_wavefunc.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <omp.h>

#include <functional>

#include "utils.h"
#include "integrate.h"


ShWavefunc::ShWavefunc(cdouble *data, const ShGrid *grid, const int m):
    data(data),
    grid(grid),
    m(m),
    data_own(false) {
    if (data == nullptr) {
        this->data = new cdouble[grid->size()]();
        data_own = true;
    }
}

ShWavefunc::~ShWavefunc() {
    if (data_own) {
        delete[] data;
    }
}


void ShWavefunc::copy(ShWavefunc* wf_dest) const {
#pragma omp parallel for
    for (int i = 0; i < grid->size(); ++i) {
        wf_dest->data[i] = data[i];
	}
}

typedef double (*func_wf_t)(ShWavefunc const* wf, int ir, int il);
typedef cdouble (*func_complex_wf_t)(ShWavefunc const* wf, int ir, int il);

/*
inline double ShWavefunc_integrate_o3(ShWavefunc const* wf, func_wf_t func, int l_max) {
	double res = 0.0;
#pragma omp parallel for reduction(+:res)
	for (int il = 0; il < l_max; ++il) {
		int ir = 0;
		{
			//res += 6*func(wf, ir, il);
			res += 4*func(wf, ir, il) + func(wf, ir+1, il);
		}
		for (ir = 2; ir < wf->grid->n[iR]-1; ir+=2) {
			res += func(wf, ir-1, il) + 4*func(wf, ir, il) + func(wf, ir+1, il);
		}

		if (ir != wf->grid->n[iR]-2) {
			ir = wf->grid->n[iR]-2;
			res += (func(wf, ir, il) + func(wf, ir+1, il))*3*0.5;
		}
	}
	return res*wf->grid->d[iR]/3;
}

inline double ShWavefunc_integrate_o4(ShWavefunc const* wf, func_wf_t func, int l_max) {
	double res = 0.0;
#pragma omp parallel for reduction(+:res)
	for (int il = 0; il < l_max; ++il) {
		int ir = 0;
		{
			int ir = 1;
			res += 32*func(wf, ir-1, il) + 12*func(wf, ir, il) + 32*func(wf, ir+1, il) + 7*func(wf, ir+2, il);
		}
		for (ir = 5; ir < wf->grid->n[iR]-3; ir+=4) {
			res += 7*func(wf, ir-2, il) + 32*func(wf, ir-1, il) + 12*func(wf, ir, il) + 32*func(wf, ir+1, il) + 7*func(wf, ir+2, il);
		}

		for (ir -= 2; ir < wf->grid->n[iR]-1; ir++) {
			res += (func(wf, ir, il) + func(wf, ir+1, il))*0.5*90.0/4.0;
		}
	}
	return 4.0*res*wf->grid->d[iR]/90.0;
}

inline double ShWavefunc_integrate_r2(ShWavefunc const* wf, func_wf_t func, int l_max, int Z) {
	double res = 0.0;

#pragma omp parallel for reduction(+:res)
	for (int il = 0; il < l_max; ++il) {
		int ir = 0;
		{
			if (il == 0) {
				double c = -Z*4*wf->grid->d[iR];
				res += ((16*c + 6)*func(wf, ir, il) + 2*c*func(wf, ir+1, il))/(1+3*c);
			} else {
				res += 6*func(wf, ir, il);
			}
		}
		for (ir = 2; ir < wf->grid->n[iR]-1; ir+=2) {
			res += func(wf, ir-1, il) + 4*func(wf, ir, il) + func(wf, ir+1, il);
		}

		if (ir != wf->grid->n[iR]-2) {
			ir = wf->grid->n[iR]-2;
			res += (func(wf, ir, il) + func(wf, ir+1, il))*3*0.5;
		}
	}
	return res*wf->grid->d[iR]/3;
}
*/

cdouble ShWavefunc::operator*(ShWavefunc const& other) const {
    return integrate<cdouble>([this, &other](ShWavefunc const* wf, int ir, int il) -> cdouble {
				return (*this)(ir, il)*conj(other(ir, il));
			}, min(grid->n[iL], other.grid->n[iL]));
}

void ShWavefunc::exclude(ShWavefunc const& other) {
	auto proj = (*this)*other / other.norm();

#pragma omp parallel for collapse(2)
	for (int il = 0; il < grid->n[iL]; il++) {
		for (int ir = 0; ir < grid->n[iR]; ir++) {
			(*this)(ir, il) -= other(ir, il)*proj;
		}
	}
}

void ShWavefunc::ort_l(int l, int n, ShWavefunc** wfs) {
	assert(n > 1);
	for (int in=1; in<n; ++in) {
		assert(wfs[in-1]->m == wfs[in]->m);
	}

    ShGrid const* grid = wfs[0]->grid;

	cdouble proj[n];
	double norm[n];

	for (int in=0; in<n; ++in) {
		for (int ip=0; ip<in; ++ip) {
            proj[ip] = ((*wfs[ip])*(*wfs[in])) / norm[ip];
		}

		for (int ip=0; ip<in; ++ip) {
            cdouble* psi = &(*wfs[in])(0, l);
			for (int ir=0; ir<grid->n[iR]; ++ir) {
				psi[ir] -= proj[ip]*(*wfs[ip])(ir, l);
			}
		}

        norm[in] = wfs[in]->norm(NULL);
	}
}

void ShWavefunc::n_sp(SpGrid const* grid, double* n, ylm_cache_t const* ylm_cache) const {
#pragma omp parallel for collapse(2)
	for (int ir = 0; ir < grid->n[iR]; ++ir) {
		for (int ic = 0; ic < grid->n[iC]; ++ic) {
			int index[3] = {ir, ic, 0};
            cdouble const psi = get_sp(grid, index, ylm_cache);
			n[ir + ic*grid->n[iR]] = pow(creal(psi), 2) + pow(cimag(psi), 2);
		}
	}
}

double ShWavefunc::norm(sh_f mask) const {
	if (mask == nullptr) {
        return  integrate<double>([](ShWavefunc const* wf, int ir, int il) {
					return wf->abs_2(ir, il);
				}, grid->n[iL]);
	} else {
        return integrate<double>([mask](ShWavefunc const* wf, int ir, int il) {
			return wf->abs_2(ir, il)*mask(wf->grid, ir, il, wf->m);
		}, grid->n[iL]);
	}
}

void ShWavefunc::normalize() {
    double norm = this->norm(NULL);
#pragma omp parallel for
    for (int i = 0; i < grid->size(); ++i) {
        data[i] /= sqrt(norm);
	}
}

void ShWavefunc::print() const {
    for (int ir = 0; ir < grid->n[iR]; ++ir) {
        double const r = grid->r(ir);
		double res = 0.0;
        for (int il = 0; il < grid->n[iL]; ++il) {
            res += pow(cabs((*this)(ir, il)), 2);
		}
		printf("%f ", res/(r*r));
	}
	printf("\n");
}

// <psi|U(r)cos(\theta)|psi>
double ShWavefunc::cos(sh_f func) const {
    return 2*integrate<double>([func](ShWavefunc const* wf, int ir, int il) -> double {
		return clm(il, wf->m)*creal((*wf)(ir, il)*conj((*wf)(ir, il+1)))*func(wf->grid, ir, il, wf->m);
	}, grid->n[iL]-1);
}

double ShWavefunc::cos_r2(sh_f U, int Z) const {
    return 2*integrate<double>([U](ShWavefunc const* wf, int ir, int il) -> double {
		return clm(il, wf->m)*creal((*wf)(ir, il)*conj((*wf)(ir, il+1)))*U(wf->grid, ir, il, wf->m);
    }, grid->n[iL]-1);
}

void ShWavefunc::cos_r(sh_f U, double* res) const {
    for (int ir = 0; ir < grid->n[iR]; ++ir) {
		res[ir] = 0.0;
        for (int il = 0; il < grid->n[iL]-1; ++il) {
            res[ir] += 2*clm(il, m)*creal((*this)(ir, il)*conj((*this)(ir, il+1)))*U(grid, ir, il, m);
		}
	}
}

double ShWavefunc::z() const {
    return cos([](ShGrid const* grid, int ir, int il, int im) {
			return grid->r(ir);
	});
}

cdouble ShWavefunc::get_sp(SpGrid const* grid, int i[3], ylm_cache_t const* ylm_cache) const {
	cdouble res = 0.0;
    for (int il = 0; il < this->grid->n[iL]; ++il) {
        int const l = this->grid->l(il);
        res += (*this)(i[iR], il)*(*ylm_cache)(l, m, i[iC]);
	}

    double r = grid->r(i[iR]);
	return res/r;
}

void ShWavefunc::random_l(int l) {
    assert(l >= 0 && l < grid->n[iL]);

    for (int il=0; il<grid->n[iL]; ++il) {
        for (int ir=0; ir<grid->n[iR]; ++ir) {
            (*this)(ir, il) = 0.0;
		}
	}

	{
		int il = l;
        cdouble* psi = &(*this)(0, il);

        for (int ir=0; ir<grid->n[iR]; ++ir) {
            double const r = grid->r(ir);
			psi[ir] = (double)rand()/(double)RAND_MAX*r*exp(-r/(2*l+1));
		}

		for (int i=0; i<10; ++i) {
            for (int ir=1; ir<grid->n[iR]-1; ++ir) {
				psi[ir] = (psi[ir-1] + psi[ir] + psi[ir+1])/3.0;
			}
		}
	}
}
