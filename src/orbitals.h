#pragma once

#include "sh_wavefunc.h"
#include "atom.h"

#include <mpi.h>
#include <array>
#include <optional>


template <typename T>
inline decltype(MPI_DOUBLE) getMpiType();

template <>
inline decltype(MPI_DOUBLE) getMpiType<double>() {
	return MPI_DOUBLE;
}

template <>
inline decltype(MPI_DOUBLE) getMpiType<cdouble>() {
	return MPI_C_DOUBLE_COMPLEX;
}

/*!
 * \brief Орбитали Кона-Шэма
 * */
template <typename Grid>
class Orbitals {
public:
    Orbitals(Atom const& atom, Grid const& grid, MPI_Comm mpi_comm, int const orbsRank[] = nullptr);
    ~Orbitals();

    void init();
	void init_shell(int shell);

	Orbitals* copy() const;
	void copy(Orbitals& dest) const;

    /*!
     *  \brief Init state [MPI support]
     *  \param data[in] is array[Ne, Nr, l_max] only for root
     */
    void setInitState(cdouble* data, int Nr, int Nl);

	template <typename T>
	void calc_array(std::function<T (Wavefunc<Grid> const*, int ie)> func, T res[]) const;

	template <typename T>
	void calc_array(std::function<T (Wavefunc<Grid> const*)> func, T res[]) const;

	double calc_sum(std::function<double (Wavefunc<Grid> const*, int ie)> func) const;
	double calc_sum(std::function<double (Wavefunc<Grid> const*)> func) const;

    /*!
     * \brief data[:] = value [MPI support]
     */
	void set(cdouble value);

	void mean(Orbitals const& other);

    /*!
     * \brief [MPI support]
     */
    double norm(typename Wavefunc<Grid>::sh_f mask = nullptr) const;
    void norm_ne(double* n, typename Wavefunc<Grid>::sh_f mask = nullptr) const;

    void prod_ne(Orbitals const& orbs, cdouble *n) const;

    /*!
     * \brief [MPI support]
     */
    void normalize(bool activeOrbs[] = nullptr, double norm[] = nullptr);

    /*!
     * \brief [Расчет дипольного момента.
	 * MPI support]
     */
    double z(typename Wavefunc<Grid>::sh_f mask = nullptr) const;
    double z2(typename Wavefunc<Grid>::sh_f mask = nullptr) const;

    /*!
     * \brief [Расчет дипольного момента для каждой орбитали.
	 * MPI support]
     */
	void z_ne(double* z, typename Wavefunc<Grid>::sh_f mask = nullptr) const;
	void z2_ne(double* z2, typename Wavefunc<Grid>::sh_f mask = nullptr) const;

    /*!
     * Электронная плотность
     * \param ne[in] is count Kohn's orbitals
     * \param wf[in] is wavefunction of Kohn's orbitals
     * \param i is sphere index \f${i_r, i_\Theta, i_\phi}\f$
     * \brief [MPI not support]
     */
    void n_sp(SpGrid const& grid, double* n, double* n_tmp, YlmCache const* ylm_cache, std::optional<int> Lmax = std::nullopt) const;
    double  n(SpGrid const* grid, int i[2], YlmCache const* ylm_cache) const;
    void n_l0(double* n, double* n_tmp) const;

    double cos(typename Wavefunc<Grid>::sh_f U) const;
	double cos(double* U) const {
		auto func = [&](int ir, int il, int m) -> double {
			return U[ir];
		};

		return cos(func);
	}

	void collect(cdouble* dest, int Nr = -1, int Nl = -1) const;

    void ort();

#ifdef _MPI
	bool is_root() const { return mpi_comm == MPI_COMM_NULL || mpi_rank == 0; }
#endif

    Atom const& atom;
    Grid const grid;
    Wavefunc<Grid>** wf; //!< волновые функции орбиталей
    cdouble* data; //!< raw data[ir + il*nr + ie*nr*nl]

    int mpi_rank;
#ifdef _MPI
    MPI_Comm mpi_comm;
    MPI_Comm spin_comm;
    Wavefunc<Grid>* mpi_wf;

	std::vector<int> ne_rank; // number of orbital -> mpi rank
#endif
};

typedef Orbitals<ShGrid> ShOrbitals;
typedef Orbitals<ShNotEqudistantGrid> ShNeOrbitals;
