#include "orbitals.h"

#include <string.h>
#include <stdio.h>


template <typename Grid>
Orbitals<Grid>::Orbitals(Atom const& atom, Grid const& grid, MPI_Comm mpi_comm, int const orbsRank[]):
    atom(atom),
    grid(grid),
	data(nullptr),
	mpi_rank(0)
{
    wf = new Wavefunc<Grid>*[atom.countOrbs];

#ifdef _MPI
	this->mpi_comm = mpi_comm;
    spin_comm = MPI_COMM_NULL;

	if (mpi_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(mpi_comm, &mpi_rank);

		int mpiCountOrbs = 0;
		ne_rank.resize(atom.countOrbs);

		if (orbsRank == nullptr) {
			for (int ie = 0; ie < atom.countOrbs; ++ie) {
				ne_rank[ie] = ie;
			}

			mpiCountOrbs = 1;
		} else {
			for (int ie = 0; ie < atom.countOrbs; ++ie) {
				ne_rank[ie] = orbsRank[ie];

				if (orbsRank[ie] == mpi_rank) {
					mpiCountOrbs += 1;
				}
			}
		}

		if (mpiCountOrbs != 0) {
			data = new cdouble[grid.size()*mpiCountOrbs]();
		}

		int iOrbs = 0;
		for (int ie = 0; ie < atom.countOrbs; ++ie) {
			if (ne_rank[ie] == mpi_rank) {
				wf[ie] = new Wavefunc<Grid>(&data[grid.size()*iOrbs], grid, atom.orbs[ie].m);
				iOrbs += 1;
			} else {
				wf[ie] = nullptr;
			}
		}

		if (atom.isSpinPolarized()) {
			MPI_Comm_split(mpi_comm, atom.orbs[mpi_rank].s, mpi_rank, &spin_comm);
		}
	} else
#endif
	{
        data = new cdouble[grid.size()*atom.countOrbs]();
        for (int ie = 0; ie < atom.countOrbs; ++ie) {
            wf[ie] = new Wavefunc<Grid>(&data[grid.size()*ie], grid, atom.orbs[ie].m);
		}
	}
}

template <typename Grid>
void Orbitals<Grid>::init() {
    for (int ie = 0; ie < atom.countOrbs; ++ie) {
        if (wf[ie] != nullptr) {
            wf[ie]->m = atom.orbs[ie].m;
            wf[ie]->random_l(atom.orbs[ie].l);
        }
    }
}

template <typename Grid>
void Orbitals<Grid>::init_shell(int shell) {
    for (int ie = 0; ie < atom.countOrbs; ++ie) {
        if (atom.orbs[ie].shell == shell && wf[ie] != nullptr) {
            wf[ie]->m = atom.orbs[ie].m;
            wf[ie]->random_l(atom.orbs[ie].l);
        }
    }
}

template <typename Grid>
Orbitals<Grid>* Orbitals<Grid>::copy() const {
	auto res = new Orbitals<Grid>(atom, grid, this->mpi_comm, &ne_rank[0]);
	copy(*res);
	return res;
}

template <typename Grid>
void Orbitals<Grid>::copy(Orbitals& dest) const {
    for (int ie=0; ie<atom.countOrbs; ++ie) {
        if (wf[ie] != nullptr) {
			wf[ie]->copy(dest.wf[ie]);
        }
    }
}

template <typename Grid>
void Orbitals<Grid>::mean(Orbitals const& other) {
    for (int ie=0; ie<atom.countOrbs; ++ie) {
        if (wf[ie] != nullptr) {
			wf[ie]->mean(other.wf[ie]);
        }
    }
}

template <typename Grid>
void Orbitals<Grid>::set(cdouble value) {
	for (int ie = 0; ie < atom.countOrbs; ++ie) {
		if (wf[ie] != nullptr) {
			wf[ie]->set(value);
		}
	}
}

template <typename Grid>
void Orbitals<Grid>::setInitState(cdouble* data, int Nr, int Nl) {
	int Nr_send = std::min(Nr, grid.n[iR]);

	if (grid.n[iR] > Nr) {
		this->set(0.0);
	}

	for (int ie = 0; ie < atom.countOrbs; ++ie) {
		if (mpi_rank == 0 && wf[ie] != nullptr) {
			for (int l = 0; l < Nl; ++l) {
				memcpy(&(*wf[ie])(0, l), &data[ie*Nr*Nl + Nr*l], Nr_send*sizeof(cdouble));
			}
		}
#ifdef _MPI
		else if (mpi_rank == 0 || wf[ie] != nullptr) {
			for (int l = 0; l < Nl; ++l) {
				if (mpi_rank == 0) {
					MPI_Send(&data[ie*Nr*Nl + Nr*l], Nr_send, MPI_C_DOUBLE_COMPLEX, ne_rank[ie], 0, mpi_comm);
				} else {
					MPI_Recv(&(*wf[ie])(0, l), Nr_send, MPI_C_DOUBLE_COMPLEX, 0, 0, mpi_comm, MPI_STATUS_IGNORE);
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

template <typename Grid>
Orbitals<Grid>::~Orbitals() {
    for (int ie = 0; ie < atom.countOrbs; ++ie) {
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
void Orbitals<Grid>::calc_array(std::function<T (Wavefunc<Grid> const*, int ie)> func, T res[]) const {
	for (int ie=0; ie<atom.countOrbs; ++ie) {
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
void Orbitals<Grid>::calc_array(std::function<T (Wavefunc<Grid> const*)> func, T res[]) const {
	calc_array<T>([&](auto wf, int ie)->T {
			return func(wf)*atom.orbs[ie].countElectrons;
	}, res);
}

template <typename Grid>
double Orbitals<Grid>::calc_sum(std::function<double (Wavefunc<Grid> const*, int ie)> func) const {
    double res = 0.0;
    double local_res = 0.0;

    for (int ie=0; ie<atom.countOrbs; ++ie) {
        if (wf[ie] != nullptr) {
			local_res += func(wf[ie], ie);
        }
    }

#ifdef _MPI
    if (mpi_comm != MPI_COMM_NULL) {
        MPI_Reduce(&local_res, &res, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
    } else
#endif
    {
        res = local_res;
    }

    return res;
}

template <typename Grid>
double Orbitals<Grid>::calc_sum(std::function<double (Wavefunc<Grid> const*)> func) const {
    return calc_sum([&](auto wf, int ie) -> double {
			return func(wf)*atom.orbs[ie].countElectrons;
	});
}

template <typename Grid>
double Orbitals<Grid>::norm(typename Wavefunc<Grid>::sh_f mask) const {
	return calc_sum([&](auto wf) -> double {
			return wf->norm(mask);
	});
}

template <typename Grid>
double Orbitals<Grid>::z(typename Wavefunc<Grid>::sh_f mask) const {
	return calc_sum([&](auto wf) -> double {
			return wf->z(mask);
	});
}

template <typename Grid>
void Orbitals<Grid>::z_ne(double* z, typename Wavefunc<Grid>::sh_f mask) const {
	calc_array<double>([&](auto wf) -> double {
			return wf->z(mask);
	}, z);
}

template <typename Grid>
void Orbitals<Grid>::norm_ne(double* n, typename Wavefunc<Grid>::sh_f mask) const {
	calc_array<double>([&](auto wf) -> double {
			return wf->norm(mask);
	}, n);
}

template <typename Grid>
void Orbitals<Grid>::prod_ne(const Orbitals &orbs, cdouble *res) const {
	return calc_array<cdouble>([&](auto wf, int ie) -> cdouble {
			return (*wf)*(*orbs.wf[ie]);
			}, res);
}

template <typename Grid>
void Orbitals<Grid>::normalize(bool activeOrbs[]) {
	for (int ie=0; ie<atom.countOrbs; ++ie) {
		if (wf[ie] != nullptr && (activeOrbs == nullptr || activeOrbs[ie])) {
			wf[ie]->normalize();
		}
	}
}

template <typename Grid>
double Orbitals<Grid>::cos(typename Wavefunc<Grid>::sh_f U) const {
	return calc_sum([&](auto wf) -> double {
			return wf->cos(U);
	});
}

template <typename Grid>
void Orbitals<Grid>::n_sp(SpGrid const& grid, double* n, double* n_tmp, YlmCache const* ylm_cache) const {
#ifdef _MPI
	if (mpi_comm == MPI_COMM_NULL)
#endif
	{
		n_tmp = n;
	}

#pragma omp parallel for collapse(2)
	for (int ic = 0; ic < grid.n[iC]; ++ic) {
		for (int ir = 0; ir < grid.n[iR]; ++ir) {
			double res = 0.0;

			for (int ie = 0; ie < atom.countOrbs; ++ie) {
				if (wf[ie] != nullptr) {
					int index[3] = {ir, ic, 0};
					cdouble const psi = wf[ie]->get_sp(grid, index, ylm_cache);
					res += (pow(creal(psi), 2) + pow(cimag(psi), 2))*atom.orbs[ie].countElectrons;
				}
			}

			n_tmp[ir + ic*grid.n[iR]] = res;
		}
	}

#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
		MPI_Reduce(n_tmp, n, grid.n[iR]*grid.n[iC], MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
	}
#endif
}

template <typename Grid>
void Orbitals<Grid>::n_l0(double* n, double* n_tmp) const {
#ifdef _MPI
	if (mpi_comm == MPI_COMM_NULL)
#endif
	{
		n_tmp = n;
	}

#pragma omp parallel for
	for (int ir = 0; ir < grid.n[iR]; ++ir) {
		n_tmp[ir] = 0;
		for (int ie = 0; ie < atom.countOrbs; ++ie) {
			if (wf[ie] != nullptr) {
				double res = 0.0;
				for (int il = wf[ie]->m; il < grid.n[iL]; ++il) {
					res += wf[ie]->abs_2(ir, il);
				}
				n_tmp[ir] += res*atom.orbs[ie].countElectrons / (pow(grid.r(ir), 2)*4*M_PI);
			}
		}
	}

#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
		MPI_Reduce(n_tmp, n, grid.n[iR], MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
	}
#endif
}

template <typename Grid>
double Orbitals<Grid>::n(SpGrid const* grid, int i[2], YlmCache const* ylm_cache) const {
#ifdef _MPI
	assert(mpi_comm == MPI_COMM_NULL);
#endif

	double res = 0.0;

	for (int ie = 0; ie < atom.countOrbs; ++ie) {
		int index[3] = {i[0], i[1], 0};
		cdouble const psi = wf[ie]->get_sp(grid[0], index, ylm_cache);
		res += (pow(creal(psi), 2) + pow(cimag(psi), 2))*atom.orbs[ie].countElectrons;
	}

	return res;
}

template <typename Grid>
void Orbitals<Grid>::ort() {
	int ie = 0;

	while (ie < atom.countOrbs) {
		int ne = atom.getNumberOrt(ie);

		if (ne > 1) {
			Wavefunc<Grid>::ort_l(atom.orbs[ie].l, ne, &wf[ie]);
		}

		ie += ne;
	}
}

template <typename Grid>
void Orbitals<Grid>::collect(cdouble* dest, int Nl) const {
	if (Nl == -1) {
		Nl = grid.n[iL];
	}

	int size = grid.n[iR]*Nl;

#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
		MPI_Gather(data, size, MPI_C_DOUBLE_COMPLEX, dest, size, MPI_C_DOUBLE_COMPLEX, 0, mpi_comm);
	} else
#endif
	{
		assert(false);
	}
}

template class Orbitals<ShGrid>;
template class Orbitals<ShNotEqudistantGrid>;
