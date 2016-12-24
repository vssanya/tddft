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
} sphere_wavefunc_t;

sphere_wavefunc_t* sphere_wavefunc_new(
		sh_grid_t const* grid,
		int const m
);

sphere_wavefunc_t* sphere_wavefunc_new_from(
		cdouble* data,
		sh_grid_t const* grid,
		int const m
);

inline cdouble* swf_ptr(sphere_wavefunc_t const* wf, int ir, int il) {
	return &wf->data[grid2_index(wf->grid, (int[2]){ir, il})];
}

inline cdouble const* swf_const_ptr(sphere_wavefunc_t const* wf, int ir, int il) {
	return &wf->data[grid2_index(wf->grid, (int[2]){ir, il})];
}

inline cdouble swf_get(sphere_wavefunc_t const* wf, int ir, int il) {
	return wf->data[grid2_index(wf->grid, (int[2]){ir, il})];
}

/*!
 * \return \f$\psi(r, \Omega)\f$
 * */
cdouble swf_get_sp(sphere_wavefunc_t const* wf, int i[3]);

inline void swf_set(sphere_wavefunc_t const* wf, int ir, int il, cdouble value) {
	wf->data[grid2_index(wf->grid, (int[2]){ir, il})] = value;
}

inline double swf_get_abs_2(sphere_wavefunc_t const* wf, int ir, int il) {
	cdouble const value = swf_get(wf, ir, il);
	return pow(creal(value), 2) + pow(cimag(value), 2);
}

void   sphere_wavefunc_del(sphere_wavefunc_t* wf);

double sphere_wavefunc_norm(sphere_wavefunc_t const* wf);

void   sphere_wavefunc_normalize(sphere_wavefunc_t* wf);

void   sphere_wavefunc_print(sphere_wavefunc_t const* wf);

// <psi|U(r)cos(\theta)|psi>
double sphere_wavefunc_cos(
		sphere_wavefunc_t const* wf,
		sphere_pot_t U
);

double sphere_wavefunc_z(sphere_wavefunc_t const* wf);
