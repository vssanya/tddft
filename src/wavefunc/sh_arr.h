#pragma once

#include "mpi_utils.h"
#include "atom.h"

#include "sh_2d.h"


/*!
 * \brief Array of wavefunction
 * */
template <typename Grid>
class WavefuncArray {
public:
	WavefuncArray(int N, Grid const& grid, int const m[], MPI_Comm mpi_comm, int const rank[] = nullptr);
	~WavefuncArray();

	template <typename T>
		void calc_array(std::function<T (Wavefunc<Grid> const*, int ie)> func, T res[]) const;

	template <typename T>
		void calc_array(std::function<T (Wavefunc<Grid> const*)> func, T res[]) const;

	/*!
	 * \brief data[:] = value [MPI support]
	 */
	void set(cdouble value);
	void set(int index, Wavefunc<Grid> const* wf);
	void set_all(Wavefunc<Grid> const* wf);

	/*!
	 * \brief [MPI support]
	 */
	void norm(double* n, typename Wavefunc<Grid>::sh_f mask = nullptr) const;

	/*!
	 * \brief [MPI support]
	 */
	void normalize(bool activeOrbs[] = nullptr, double norm[] = nullptr);

	/*!
	 * \brief [Расчет дипольного момента для каждой орбитали.
	 * MPI support]
	 */
	void z(double* z, typename Wavefunc<Grid>::sh_f mask = nullptr) const;
	void z2(double* z2, typename Wavefunc<Grid>::sh_f mask = nullptr) const;

	void cos(double* res, typename Wavefunc<Grid>::sh_f U) const;
	void cos(double* res, double* U) const {
		auto func = [&](int ir, int il) -> double {
			return U[ir];
		};

		return cos(res, func);
	}

	void collect(cdouble* dest, int Nr = -1, int Nl = -1) const;

#ifdef _MPI
	bool is_root() const { return mpi_comm == MPI_COMM_NULL || mpi_rank == 0; }
#endif

	int const N;
	Grid const grid;
	Wavefunc<Grid>** wf; //!< волновые функции орбиталей
	cdouble* data; //!< raw data[ir + il*nr + ie*nr*nl]

	int mpi_rank;
#ifdef _MPI
	int mpi_size;
	MPI_Comm mpi_comm;

	Wavefunc<Grid>* mpi_wf;
	int count_wf;

	std::vector<int> ne_rank; // index of wavefunction -> mpi rank
	std::vector<int> rank_count; // rank -> count of wavefunction
	std::vector<int> rank_shift;
#endif
};

typedef WavefuncArray<ShGrid> ShWavefuncArray;
typedef WavefuncArray<ShNotEqudistantGrid> ShNeWavefuncArray;
