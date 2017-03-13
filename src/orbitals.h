#pragma once

#include "sh_wavefunc.h"

#include <mpi/mpi.h>


/*!
 * \brief Орбитали Кона-Шэма
 * */
typedef struct {
	int ne; //!< число орбиталей Кона-Шэма
	sh_grid_t const* grid;
	sh_wavefunc_t** wf; //!< волновые функции орбиталей
	cdouble* data; //!< raw data[ir + il*nr + ie*nr*nl]

	MPI_Comm mpi_comm;
	int mpi_rank;
	sh_wavefunc_t* mpi_wf;
} orbitals_t;

orbitals_t* ks_orbials_new(int ne, sh_grid_t const* grid, MPI_Comm mpi_comm);
void orbitals_del(orbitals_t* orbs);

double orbitals_norm(orbitals_t const* orbs);
void orbitals_normalize(orbitals_t* orbs);

/*!
 * Электронная плотность
 * \param ne[in] is count Kohn's orbitals
 * \param wf[in] is wavefunction of Kohn's orbitals
 * \param i is sphere index \f${i_r, i_\Theta, i_\phi}\f$
 * */
double orbitals_n(orbitals_t const* orbs, sp_grid_t const* grid, int i[2], ylm_cache_t const* ylm_cache);
void orbitals_n_sp(orbitals_t const* orbs, sp_grid_t const* grid, double n[grid->n[iR]*grid->n[iC]], ylm_cache_t const* ylm_cache);

double orbitals_cos(orbitals_t const* orbs, sh_f U);
