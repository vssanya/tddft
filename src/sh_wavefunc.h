#pragma once

#include <stdbool.h>
#include <math.h>

#include "types.h"
#include "grid.h"
#include "sphere_harmonics.h"

/*!
 * \brief Волновая функция представленная в виде разложения по сферическим гармоникам
 *
 * \f[\psi(\vec{r}) = \frac{1}{r} \sum_{l=0}^{\infty} \Theta_{lm}(r) Y_{lm}(\theta, \phi) \simeq \frac{1}{r} \sum_{l=0}^{N_l - 1} \Theta_{lm}(r) Y_{lm}(\theta, \phi)\f]
 *
 * */
typedef struct {
    sh_grid_t const* grid;           

    cdouble* data; //!< data[i + l*grid->Nr] = \f$\Theta_{lm}(r_i)\f$
	bool data_own;      //!< кто выделил данные

    int m;         //!< is magnetic quantum number
} sh_wavefunc_t;

sh_wavefunc_t* sh_wavefunc_new(
		sh_grid_t const* grid,
		int const m
);

void sh_wavefunc_random_l(sh_wavefunc_t* wf, int l);

sh_wavefunc_t* sh_wavefunc_new_from(
		cdouble* data,
		sh_grid_t const* grid,
		int const m
);

inline cdouble* swf_ptr(sh_wavefunc_t const* wf, int ir, int il) {
	return &wf->data[grid2_index(wf->grid, (int[2]){ir, il})];
}

inline cdouble const* swf_const_ptr(sh_wavefunc_t const* wf, int ir, int il) {
	return &wf->data[grid2_index(wf->grid, (int[2]){ir, il})];
}

inline cdouble swf_get(sh_wavefunc_t const* wf, int ir, int il) {
	return wf->data[grid2_index(wf->grid, (int[2]){ir, il})];
}

inline void swf_set(sh_wavefunc_t* wf, int ir, int il, cdouble value) {
	wf->data[grid2_index(wf->grid, (int[2]){ir, il})] = value;
}

void sh_wavefunc_ort_l(int l, int n, sh_wavefunc_t* wfs[n]);

/*!
 * \return \f$\psi(r, \Omega)\f$
 * */
cdouble swf_get_sp(sh_wavefunc_t const* wf, sp_grid_t const* grid, int i[3]);

inline double swf_get_abs_2(sh_wavefunc_t const* wf, int ir, int il) {
	cdouble const value = swf_get(wf, ir, il);
	return pow(creal(value), 2) + pow(cimag(value), 2);
}

void   sh_wavefunc_del(sh_wavefunc_t* wf);

// \return \f$<\psi_1|\psi_2>\f$
cdouble sh_wavefunc_prod(sh_wavefunc_t const* wf1, sh_wavefunc_t const* wf2);

void sh_wavefunc_n_sp(sh_wavefunc_t const* wf, sp_grid_t const* grid, double n[grid->n[iR]*grid->n[iC]]);

double sh_wavefunc_norm(sh_wavefunc_t const* wf);

void   sh_wavefunc_normalize(sh_wavefunc_t* wf);

void   sh_wavefunc_print(sh_wavefunc_t const* wf);

// <psi|U(r)cos(\theta)|psi>
double sh_wavefunc_cos(
		sh_wavefunc_t const* wf,
		sh_f U
);

double sh_wavefunc_z(sh_wavefunc_t const* wf);
