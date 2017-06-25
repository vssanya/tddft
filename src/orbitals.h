#pragma once

#include "sh_wavefunc.h"
#include "atom.h"

#include <mpi/mpi.h>


/*!
 * \brief Орбитали Кона-Шэма
 * */
typedef struct orbitals_s {
	atom_t const* atom;
	sh_grid_t const* grid;
	sh_wavefunc_t** wf; //!< волновые функции орбиталей
	cdouble* data; //!< raw data[ir + il*nr + ie*nr*nl]

#ifdef _MPI
	MPI_Comm mpi_comm;
	int mpi_rank;
	sh_wavefunc_t* mpi_wf;
#endif
} orbitals_t;

orbitals_t* orbials_new(atom_t const* atom, sh_grid_t const* grid, MPI_Comm mpi_comm);
void orbitals_del(orbitals_t* orbs);

void orbitals_init(orbitals_t* orbs);

/*!
 *  \brief Init state [MPI support]
 *  \param data[in] is array[Ne, Nr, l_max] only for root
 */
void orbitals_set_init_state(orbitals_t* orbs, cdouble* data, int n_r, int n_l);

/*!
 * \brief [MPI support]
 */
double orbitals_norm(orbitals_t const* orbs, sh_f mask);
void orbitals_norm_ne(orbitals_t const* orbs, double n[orbs->atom->n_orbs], sh_f mask);
/*!
 * \brief [MPI support]
 */
void orbitals_normalize(orbitals_t* orbs);

/*!
 * \brief [MPI support]
 */
double orbitals_z(orbitals_t const* orbs);

/*!
 * Электронная плотность
 * \param ne[in] is count Kohn's orbitals
 * \param wf[in] is wavefunction of Kohn's orbitals
 * \param i is sphere index \f${i_r, i_\Theta, i_\phi}\f$
 * */
/*!
 * \brief [MPI not support]
 */
double orbitals_n(orbitals_t const* orbs, sp_grid_t const* grid, int i[2], ylm_cache_t const* ylm_cache);
void orbitals_n_sp(orbitals_t const* orbs, sp_grid_t const* grid, double n[grid->n[iR]*grid->n[iC]], double n_tmp[grid->n[iR]*grid->n[iC]], ylm_cache_t const* ylm_cache);
void orbitals_n_sh(orbitals_t const* orbs, double n[orbs->grid->n[iR]], double n_tmp[orbs->grid->n[iR]]);

double orbitals_cos(orbitals_t const* orbs, sh_f U);

void orbitals_ort(orbitals_t* orbs);
