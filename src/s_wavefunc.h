#pragma once

#include "s_grid.h"

#include <stdlib.h>

#include "types.h"
#include "sphere_wavefunc.h"

/*!
 * \brief Общая структура для трехмерных волновых функций
 * */
typedef struct {
	grid_t const* grid;
	cdouble* data;
} wavefunc_t;

inline cdouble wf_get(wavefunc_t const* wf, int i[3]) {
	return wf->data[grid_index(wf->grid, i)];
}

inline void wf_set(wavefunc_t const* wf, int i[3], cdouble value) {
	wf->data[grid_index(wf->grid, i)];
}

wavefunc_t* wavefunc_new(grid_t const* grid);
void        wavefunc_del(wavefunc_t* wf);

/*!
 * \brief Волновая функция записанная в сферических координатах
 *
 * \f[\psi(r, \theta, \phi)\f]
 *
 * */
typedef struct {
	wavefunc_t;
} s_wavefunc_t;

void sp_wavefunc_from_sh(s_wavefunc_t* sp_wf, sphere_wavefunc_t const* sh_wf);
