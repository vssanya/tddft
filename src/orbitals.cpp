#include "orbitals.h"

#include <string.h>
#include <stdio.h>


template <typename Grid>
Orbitals<Grid>::Orbitals(Atom const& atom, Grid const& grid, MPI_Comm mpi_comm):
    atom(atom),
    grid(grid)
{
    wf = new Wavefunc<Grid>*[atom.countOrbs];

#ifdef _MPI
	this->mpi_comm = mpi_comm;
    spin_comm = MPI_COMM_NULL;

	if (mpi_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(mpi_comm, &mpi_rank);

        data = new cdouble[grid.size()]();
        for (int ie = 0; ie < atom.countOrbs; ++ie) {
            if (ie == mpi_rank) {
                wf[ie] = new Wavefunc<Grid>(data, grid, atom.orbs[ie].m);
                mpi_wf = wf[ie];
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
Orbitals<Grid>* Orbitals<Grid>::copy() const {
	auto res = new Orbitals<Grid>(atom, grid, this->mpi_comm);
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
void Orbitals<Grid>::setInitState(cdouble* data, int Nr, int Nl) {
    assert(grid.n[iR] <= Nr);

    int const size = Nl*grid.n[iR];
#ifdef _MPI
    if (mpi_comm != MPI_COMM_NULL) {
        assert(grid.n[iR] == Nr);
        MPI_Scatter(data, size, MPI_C_DOUBLE_COMPLEX, mpi_wf->data, size, MPI_C_DOUBLE_COMPLEX, 0, mpi_comm);
    } else
#endif
    {
        if (Nr == grid.n[iR]) {
            for (int ie = 0; ie < atom.countOrbs; ++ie) {
                memcpy(wf[ie]->data, &data[ie*Nr*Nl], size*sizeof(cdouble));
            }
        } else {
            for (int ie = 0; ie < atom.countOrbs; ++ie) {
                for (int il = 0; il < Nl; ++il) {
                    memcpy(&wf[ie]->data[il*grid.n[iR]], &data[il*Nr + ie*Nr*Nl], size*sizeof(cdouble));
                }
            }
        }
    }
}

template <typename Grid>
Orbitals<Grid>::~Orbitals() {
    for (int ie = 0; ie < atom.countOrbs; ++ie) {
        if (wf[ie] != NULL) {
            delete wf[ie];
        }
    }

    delete[] data;
}

template <typename Grid>
double Orbitals<Grid>::norm(typename Wavefunc<Grid>::sh_f mask) const {
    double res = 0.0;
    double local_res = 0.0;

    for (int ie=0; ie<atom.countOrbs; ++ie) {
        if (wf[ie] != nullptr) {
            local_res += wf[ie]->norm(mask)*atom.orbs[ie].countElectrons;
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
double Orbitals<Grid>::z(typename Wavefunc<Grid>::sh_f mask) const {
    double res = 0.0;
    double local_res = 0.0;

    for (int ie=0; ie<atom.countOrbs; ++ie) {
        if (wf[ie] != nullptr) {
            local_res += wf[ie]->z(mask)*atom.orbs[ie].countElectrons;
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
void Orbitals<Grid>::z_ne(double* z, typename Wavefunc<Grid>::sh_f mask) const {
#ifdef _MPI
    if (mpi_comm != MPI_COMM_NULL) {
		double local_res = mpi_wf->z(mask)*atom.orbs[mpi_rank].countElectrons;
		MPI_Gather(&local_res, 1, MPI_DOUBLE, z, 1, MPI_DOUBLE, 0, mpi_comm);
	} else
#endif
	{
		for (int ie=0; ie<atom.countOrbs; ++ie) {
			z[ie] = wf[ie]->z(mask)*atom.orbs[ie].countElectrons;
		}
	}
}

template <typename Grid>
void Orbitals<Grid>::norm_ne(double* n, typename Wavefunc<Grid>::sh_f mask) const {
#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
		double n_local = mpi_wf->norm(mask)*atom.orbs[mpi_rank].countElectrons;
		MPI_Gather(&n_local, 1, MPI_DOUBLE, n, 1, MPI_DOUBLE, 0, mpi_comm);
	} else
#endif
	{
		for (int ie=0; ie<atom.countOrbs; ++ie) {
			n[ie] = wf[ie]->norm(mask);//*atom->n_e[ie];
		}
	}
}

template <typename Grid>
void Orbitals<Grid>::prod_ne(const Orbitals &orbs, cdouble *res) const {
#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
		cdouble res_local = (*mpi_wf)*(*orbs.mpi_wf);
		MPI_Gather(&res_local, 1, MPI_DOUBLE, res, 1, MPI_DOUBLE, 0, mpi_comm);
	} else
#endif
	{
		for (int ie=0; ie<atom.countOrbs; ++ie) {
			res[ie] = (*wf[ie])*(*orbs.wf[ie]);
		}
	}
}

template <typename Grid>
void Orbitals<Grid>::normalize() {
	for (int ie=0; ie<atom.countOrbs; ++ie) {
		if (wf[ie] != nullptr) {
			wf[ie]->normalize();
		}
	}
}

template <typename Grid>
double Orbitals<Grid>::cos(typename Wavefunc<Grid>::sh_f U) const {
	double res = 0.0;
	double res_local = 0.0;

	for (int ie=0; ie<atom.countOrbs; ++ie) {
		if (wf[ie] != nullptr) {
			res_local += wf[ie]->cos(U)*atom.orbs[ie].countElectrons;
		}
	}

#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
		MPI_Reduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
	} else
#endif
	{
		res = res_local;
	}

	return res;
}

template <typename Grid>
void Orbitals<Grid>::n_sp(SpGrid const& grid, double* n, double* n_tmp, YlmCache const* ylm_cache) const {
#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
#pragma omp parallel for collapse(2)
		for (int ic = 0; ic < grid.n[iC]; ++ic) {
			for (int ir = 0; ir < grid.n[iR]; ++ir) {
				int index[3] = {ir, ic, 0};
				cdouble const psi = mpi_wf->get_sp(grid, index, ylm_cache);
				n_tmp[ir + ic*grid.n[iR]] = (pow(creal(psi), 2) + pow(cimag(psi), 2))*atom.orbs[mpi_rank].countElectrons;
			}
		}

		MPI_Reduce(n_tmp, n, grid.n[iR]*grid.n[iC], MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
	} else
#endif
	{
#pragma omp parallel
		{
#pragma omp for collapse(2)
			for (int ic = 0; ic < grid.n[iC]; ++ic) {
				for (int ir = 0; ir < grid.n[iR]; ++ir) {
					n[ir + ic*grid.n[iR]] = 0.0;
					for (int ie = 0; ie < atom.countOrbs; ++ie) {
						int index[3] = {ir, ic, 0};
						cdouble const psi = wf[ie]->get_sp(grid, index, ylm_cache);
						n[ir + ic*grid.n[iR]] += (pow(creal(psi), 2) + pow(cimag(psi), 2))*atom.orbs[ie].countElectrons;
					}
				}
			}
		}
	}
}

template <typename Grid>
void Orbitals<Grid>::n_l0(double* n, double* n_tmp) const {
#ifdef _MPI
	if (mpi_comm != MPI_COMM_NULL) {
#pragma omp parallel for
		for (int ir = 0; ir < grid.n[iR]; ++ir) {
			n_tmp[ir] = 0;
			for (int il = 0; il < grid.n[iL]; ++il) {
				n_tmp[ir] += mpi_wf->abs_2(ir, il);
			}
			n_tmp[ir] *= atom.orbs[mpi_rank].countElectrons / (pow(grid.r(ir), 2)*4*M_PI);
		}

		MPI_Reduce(n_tmp, n, grid.n[iR], MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
	} else
#endif
	{
#pragma omp parallel for
		for (int ir = 0; ir < grid.n[iR]; ++ir) {
			n[ir] = 0.0;
			for (int ie = 0; ie < atom.countOrbs; ++ie) {
				double res = 0.0;
				for (int il = 0; il < grid.n[iL]; ++il) {
					res += wf[ie]->abs_2(ir, il);
				}
				n[ir] += res*atom.orbs[ie].countElectrons / (pow(grid.r(ir), 2)*4*M_PI);
			}
		}
	}
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

template class Orbitals<ShGrid>;
template class Orbitals<ShNotEqudistantGrid>;
