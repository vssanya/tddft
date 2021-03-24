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
    spin_comm = MPI_COMM_NULL;

	if (mpi_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(mpi_comm, &mpi_rank);

		int mpiCount = 0;
		ne_rank.resize(N);

		if (rank == nullptr) {
			for (int ie = 0; ie < N; ++ie) {
				ne_rank[ie] = ie;
			}

			mpiCount = 1;
		} else {
			for (int ie = 0; ie < N; ++ie) {
				ne_rank[ie] = rank[ie];

				if (rank[ie] == mpi_rank) {
					mpiCount += 1;
				}
			}
		}

		if (mpiCount != 0) {
			data = new cdouble[grid.size()*mpiCount]();
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

			wf[ie] = new Wavefunc<Grid>(&data[grid.size()*ie], grid, m[ie]);
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

template <typename Grid> template <typename T>
void WavefuncArray<Grid>::calc_array(std::function<T (Wavefunc<Grid> const*, int ie)> func, T res[]) const {
	for (int ie=0; ie<N; ++ie) {
		if (wf[ie] != nullptr) {
			T res_local = func(wf[ie], ie);
			if (mpi_rank == 0) {
				res[ie] = res_local;
			}
#ifdef _MPI
			else {
				MPI_Send(&res_local, 1, getMpiType<T>(), 0, 0, mpi_comm);
			}
#endif
		}
#ifdef _MPI
		else if (mpi_rank == 0) {
			MPI_Recv(&res[ie], 1, getMpiType<T>(), ne_rank[ie], 0, mpi_comm, MPI_STATUS_IGNORE);
		}
#endif
	}

#ifdef _MPI
	MPI_Barrier(mpi_comm);
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
template class WavefuncArray<ShNotEqudistantGrid>;
