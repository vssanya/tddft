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
struct sh_wavefunc_t {
  sh_grid_t const* grid;

  cdouble* data; //!< data[i + l*grid->Nr] = \f$\Theta_{lm}(r_i)\f$
  bool data_own; //!< кто выделил данные

  int m;         //!< is magnetic quantum number

#ifdef __cplusplus
  inline
  cdouble& operator() (int ir, int il) {
	  return data[ir + il*grid->n[iR]];
  }

  inline
  cdouble operator() (int ir, int il) const {
	  return data[ir + il*grid->n[iR]];
  }

  cdouble operator*(sh_wavefunc_t const& other) const;
#endif
};

#ifndef __cplusplus
typedef struct sh_wavefunc_t sh_wavefunc_t;
#endif

sh_wavefunc_t* sh_wavefunc_new(
		sh_grid_t const* grid,
		int const m
);

#ifdef __cplusplus
extern "C" {
#endif

void sh_wavefunc_random_l(sh_wavefunc_t* wf, int l);

sh_wavefunc_t* sh_wavefunc_new_from(
		cdouble* data,
		sh_grid_t const* grid,
		int const m
);

void sh_wavefunc_copy(sh_wavefunc_t const* wf_src, sh_wavefunc_t* wf_dest);

inline cdouble* swf_ptr(sh_wavefunc_t const* wf, int ir, int il) {
	int index[2] = {ir, il};
	return &wf->data[grid2_index(wf->grid, index)];
}

inline cdouble const* swf_const_ptr(sh_wavefunc_t const* wf, int ir, int il) {
	int index[2] = {ir, il};
	return &wf->data[grid2_index(wf->grid, index)];
}

inline cdouble swf_get(sh_wavefunc_t const* wf, int ir, int il) {
	int index[2] = {ir, il};
	return wf->data[grid2_index(wf->grid, index)];
}

inline void swf_set(sh_wavefunc_t* wf, int ir, int il, cdouble value) {
	int index[2] = {ir, il};
	wf->data[grid2_index(wf->grid, index)] = value;
}

void sh_wavefunc_ort_l(int l, int n, sh_wavefunc_t** wfs);

/*!
 * \return \f$\psi(r, \Omega)\f$
 * */
cdouble swf_get_sp(sh_wavefunc_t const* wf, sp_grid_t const* grid, int i[3], ylm_cache_t const* ylm_cache);

double swf_get_abs_2(sh_wavefunc_t const* wf, int ir, int il);

void   sh_wavefunc_del(sh_wavefunc_t* wf);

// \return \f$<\psi_1|\psi_2>\f$
cdouble sh_wavefunc_prod(sh_wavefunc_t const* wf1, sh_wavefunc_t const* wf2);

void sh_wavefunc_n_sp(sh_wavefunc_t const* wf, sp_grid_t const* grid, double* n, ylm_cache_t const* ylm_cache);

double sh_wavefunc_norm(sh_wavefunc_t const* wf, sh_f mask);
void   sh_wavefunc_normalize(sh_wavefunc_t* wf);

void   sh_wavefunc_print(sh_wavefunc_t const* wf);

// <psi|U(r)cos(\theta)|psi>
double sh_wavefunc_cos(
		sh_wavefunc_t const* wf,
		sh_f U
);
double sh_wavefunc_cos_r2(sh_wavefunc_t const* wf, sh_f U, int Z);
void sh_wavefunc_cos_r(sh_wavefunc_t const* wf, sh_f U, double* res);

double sh_wavefunc_z(sh_wavefunc_t const* wf);

#ifdef __cplusplus
}
#endif
