#pragma once

#include "sh_wavefunc.h"

/*!
 * \brief Орбитали Кона-Шэма
 * */
typedef struct {
	int ne; //!< число орбиталей Кона-Шэма
	sh_grid_t const* grid;
	sh_wavefunc_t** wf; //!< волновые функции орбиталей
	cdouble* data; //!< raw data[ir + il*nr + ie*nr*nl]
} orbitals_t;

orbitals_t* ks_orbials_new(int ne, sh_grid_t const* grid);
void orbitals_del(orbitals_t* orbs);

double orbitals_norm(orbitals_t const* orbs);
void orbitals_normalize(orbitals_t* orbs);

/*!
 * Электронная плотность
 * \param ne[in] is count Kohn's orbitals
 * \param wf[in] is wavefunction of Kohn's orbitals
 * \param i is sphere index \f${i_r, i_\Theta, i_\phi}\f$
 * */
double orbitals_n(orbitals_t const* orbs, sp_grid_t const* grid, int i[2]);

double orbitals_cos(orbitals_t const* orbs, sh_f U);
