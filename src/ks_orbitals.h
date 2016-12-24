#pragma once

#include "sphere_wavefunc.h"

/*!
 * \brief Орбитали Кона-Шэма
 * */
typedef struct {
	int ne; //!< число орбиталей Кона-Шэма
	sphere_wavefunc_t** wf; //!< волновые функции орбиталей
	cdouble* data; //!< raw data[ir + il*nr + ie*nr*nl]
} ks_orbitals_t;

ks_orbitals_t* ks_orbials_new(int ne, sh_grid_t const* grid);

/*!
 * Электронная плотность
 * \param ne[in] is count Kohn's orbitals
 * \param wf[in] is wavefunction of Kohn's orbitals
 * \param i is sphere index \f${i_r, i_\Theta, i_\phi}\f$
 * */
double ks_orbitals_n(ks_orbitals_t const* orbs, int i[2]);
