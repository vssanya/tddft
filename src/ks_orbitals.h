#pragma once

#include "sphere_wavefunc.h"

/*!
 * \brief Орбитали Кона-Шэма
 * */
typedef struct {
	int ne; //!< число орбиталей Кона-Шэма
	sh_grid_t const* grid;
	sphere_wavefunc_t** wf; //!< волновые функции орбиталей
	cdouble* data; //!< raw data[ir + il*nr + ie*nr*nl]
} ks_orbitals_t;

ks_orbitals_t* ks_orbials_new(int ne, sh_grid_t const* grid);
void ks_orbitals_del(ks_orbitals_t* orbs);

void ks_orbitals_normilize(ks_orbitals_t* orbs);

/*!
 * Электронная плотность
 * \param ne[in] is count Kohn's orbitals
 * \param wf[in] is wavefunction of Kohn's orbitals
 * \param i is sphere index \f${i_r, i_\Theta, i_\phi}\f$
 * */
double ks_orbitals_n(ks_orbitals_t const* orbs, sp_grid_t const* grid, int i[2]);
