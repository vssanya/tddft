#pragma once

#include "sh_wavefunc.h"
#include "atom.h"

#include <mpi.h>
#include <array>


/*!
 * \brief Орбитали Кона-Шэма
 * */
class Orbitals {
public:
    Orbitals(Atom const& atom, ShGrid const* grid, MPI_Comm mpi_comm);
    ~Orbitals();

    void init();

    /*!
     *  \brief Init state [MPI support]
     *  \param data[in] is array[Ne, Nr, l_max] only for root
     */
    void setInitState(cdouble* data, int Nr, int Nl);

    /*!
     * \brief [MPI support]
     */
    double norm(sh_f mask) const;
    void norm_ne(double* n, sh_f mask) const;

    /*!
     * \brief [MPI support]
     */
    void normalize();

    /*!
     * \brief [MPI support]
     */
    double z() const;

    /*!
     * Электронная плотность
     * \param ne[in] is count Kohn's orbitals
     * \param wf[in] is wavefunction of Kohn's orbitals
     * \param i is sphere index \f${i_r, i_\Theta, i_\phi}\f$
     * */
    /*!
     * \brief [MPI not support]
     */
    double  n(SpGrid const* grid, int i[2], ylm_cache_t const* ylm_cache) const;
    void n_sp(SpGrid const* grid, double* n, double* n_tmp, ylm_cache_t const* ylm_cache) const;
    void n_l0(double* n, double* n_tmp) const;

    double cos(sh_f U) const;

    void ort();

    Atom const& atom;
    ShGrid const* grid;
    ShWavefunc** wf; //!< волновые функции орбиталей
    cdouble* data; //!< raw data[ir + il*nr + ie*nr*nl]

#ifdef _MPI
    MPI_Comm mpi_comm;
    int mpi_rank;
    ShWavefunc* mpi_wf;
#endif
};
