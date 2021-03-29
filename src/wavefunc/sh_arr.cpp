#include "sh_arr.h"

#include <string.h>
#include <stdio.h>
#include <iostream>


template <typename Grid>
WavefuncArray<Grid>::WavefuncArray(int N, Grid const& grid, int const m[], MPI_Comm mpi_comm, int const rank[]):
    N(N),
    grid(grid),
	data(nullptr),
	mpi_rank(0)
{
    wf = new Wavefunc<Grid>*[N];

#ifdef _MPI
	this->mpi_comm = mpi_comm;

	if (mpi_comm != MPI_COMM_NULL) {
		MPI_Comm_size(mpi_comm, &mpi_size);
        MPI_Comm_rank(mpi_comm, &mpi_rank);

		count_wf = 0;
		ne_rank.resize(N);

		if (rank == nullptr) {
			for (int ie = 0; ie < N; ++ie) {
				ne_rank[ie] = ie;
			}

			count_wf = 1;
		} else {
			for (int ie = 0; ie < N; ++ie) {
				ne_rank[ie] = rank[ie];

				if (rank[ie] == mpi_rank) {
					count_wf += 1;
				}
			}
		}

		if (count_wf != 0) {
			data = new cdouble[grid.size()*count_wf]();
		}

		rank_count.resize(mpi_size);
		MPI_Gather(&count_wf, 1, MPI_INT, rank_count.data(), 1, MPI_INT, 0, mpi_comm);

		rank_shift.resize(mpi_size);
		if (mpi_rank == 0) {
			rank_shift[0] = 0;
			for (int i=1; i<mpi_size; ++i) {
				rank_shift[i] = rank_shift[i-1] + rank_count[i-1];
			}
		}

		int iWf = 0;
		for (int ie = 0; ie < N; ++ie) {
			if (ne_rank[ie] == mpi_rank) {
				int m_current = 0;
				if (m != nullptr) {
					m_current = m[ie];
				}

				wf[ie] = new Wavefunc<Grid>(&data[grid.size()*iWf], grid, m_current);
				iWf += 1;
			} else {
				wf[ie] = nullptr;
			}
		}
	} else
#endif
	{
		data = new cdouble[grid.size()*N]();
		for (int ie = 0; ie < N; ++ie) {
			int m_current = 0;
			if (m != nullptr) {
				m_current = m[ie];
			}

			wf[ie] = new Wavefunc<Grid>(&data[grid.size()*ie], grid, m_current);
		}
	}
}

template <typename Grid>
void WavefuncArray<Grid>::set(cdouble value) {
	for (int ie = 0; ie < N; ++ie) {
		if (wf[ie] != nullptr) {
			wf[ie]->set(value);
		}
	}
}

template <typename Grid>
void WavefuncArray<Grid>::set(int index, Wavefunc<Grid> const* wf) {
	if (index < 0) {
		index += N;
	}

	if (this->wf[index] != nullptr) {
		wf->copy(this->wf[index]);
	}
}

template <typename Grid>
void WavefuncArray<Grid>::set_all(Wavefunc<Grid> const* wf) {
	for (int ie = 0; ie < N; ++ie) {
		set(ie, wf);
	}
}

template <typename Grid>
WavefuncArray<Grid>::~WavefuncArray() {
    for (int ie = 0; ie < N; ++ie) {
        if (wf[ie] != nullptr) {
            delete wf[ie];
        }
    }

	delete[] wf;

	if (data != nullptr) {
		delete[] data;
	}
}

// ToDo need send array to root, don't send only one element
template <typename Grid> template <typename T>
void WavefuncArray<Grid>::calc_array(std::function<T (Wavefunc<Grid> const*, int ie)> func, T res[]) const {
	T* res_local;
#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
		res_local = new T[count_wf]();
	} else 
#endif
	{
		res_local = res;
	}

	int local_i = 0;
	for (int ie=0; ie<N; ++ie) {
		if (wf[ie] != nullptr) {
			res_local[local_i] = func(wf[ie], ie);
			local_i++;
		}
	}

#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
		MPI_Gatherv(res_local, count_wf, getMpiType<T>(), res, rank_count.data(), rank_shift.data(), getMpiType<T>(), 0, mpi_comm);

		delete[] res_local;
	}
#endif
}

template <typename Grid> template <typename T>
void WavefuncArray<Grid>::calc_array(std::function<T (Wavefunc<Grid> const*)> func, T res[]) const {
	calc_array<T>([&](auto wf, int ie)->T {
			return func(wf);
	}, res);
}

template <typename Grid>
void WavefuncArray<Grid>::z(double* z, typename Wavefunc<Grid>::sh_f mask) const {
	calc_array<double>([&](auto wf) -> double {
			return wf->z(mask);
	}, z);
}

template <typename Grid>
void WavefuncArray<Grid>::z2(double* z2, typename Wavefunc<Grid>::sh_f mask) const {
	calc_array<double>([&](auto wf) -> double {
			return wf->z2(mask);
	}, z2);
}

template <typename Grid>
void WavefuncArray<Grid>::norm(double* n, typename Wavefunc<Grid>::sh_f mask) const {
	calc_array<double>([&](auto wf) -> double {
			return wf->norm(mask);
	}, n);
}

template <typename Grid>
void WavefuncArray<Grid>::normalize(bool activeOrbs[], double norm[]) {
	for (int ie=0; ie<N; ++ie) {
		if (wf[ie] != nullptr && (activeOrbs == nullptr || activeOrbs[ie])) {
			wf[ie]->normalize(norm == nullptr ? 1.0 : norm[ie]);
		}
	}
}

template <typename Grid>
void WavefuncArray<Grid>::cos(double* res, typename Wavefunc<Grid>::sh_f U) const {
	calc_array<double>([&](auto wf) -> double {
			return wf->cos(U);
	}, res);
}

template <typename Grid>
void WavefuncArray<Grid>::collect(cdouble* data, int Nr, int Nl) const {
	std::cout << "Collect Nr = " << Nr << ", Nl = " << Nl << "\n";

	if (Nl == -1) {
		Nl = grid.n[iL];
	}

	if (Nr == -1) {
		Nr = grid.n[iR];
	}

	std::cout << "Collect Nr = " << Nr << ", Nl = " << Nl << "\n";

	for (int ie = 0; ie < N; ++ie) {
		if (mpi_rank == 0 && wf[ie] != nullptr) {
			std::cout << "Copy main rank: " << ie << "\n";
			for (int l = 0; l < Nl; ++l) {
				memcpy(&data[ie*Nr*Nl + Nr*l], &(*wf[ie])(0, l), Nr*sizeof(cdouble));
			}
		}
#ifdef _MPI
		else if (mpi_rank == 0 || wf[ie] != nullptr) {
			for (int l = 0; l < Nl; ++l) {
				if (mpi_rank == 0) {
					std::cout << "Recv: " << ie << "\n";
					MPI_Recv(&data[ie*Nr*Nl + Nr*l], Nr, MPI_C_DOUBLE_COMPLEX, ne_rank[ie], 0, mpi_comm, MPI_STATUS_IGNORE);
				} else {
					std::cout << "Send: " << ie << "\n";
					MPI_Send(&(*wf[ie])(0, l), Nr, MPI_C_DOUBLE_COMPLEX, 0, 0, mpi_comm);
				}
			}
		}
#endif
	}

#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
		MPI_Barrier(mpi_comm);
	}
#endif
}

template class WavefuncArray<ShGrid>;
template void WavefuncArray<ShGrid>::calc_array<double>(std::function<double (Wavefunc<ShGrid> const*, int ie)> func, double res[]) const;
template class WavefuncArray<ShNotEqudistantGrid>;
