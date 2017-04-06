#include "orbitals.h"

#include <string.h>
#include <stdio.h>


orbitals_t* orbials_new(atom_t const* atom, sh_grid_t const* grid, MPI_Comm mpi_comm) {
	orbitals_t* orbs = malloc(sizeof(orbitals_t));

	orbs->atom = atom;
	orbs->grid = grid;

	orbs->wf = malloc(sizeof(sh_wavefunc_t*)*atom->n_orbs);

#ifdef _MPI
	orbs->mpi_comm = mpi_comm;

	if (mpi_comm != MPI_COMM_NULL) {
		MPI_Comm_rank(orbs->mpi_comm, &orbs->mpi_rank);

		orbs->data = malloc(grid2_size(grid)*sizeof(cdouble));
		for (int ie = 0; ie < atom->n_orbs; ++ie) {
			if (ie == orbs->mpi_rank) {
				orbs->wf[ie] = sh_wavefunc_new_from(orbs->data, grid, atom->m[ie]);
				orbs->mpi_wf = orbs->wf[ie];
			} else {
				orbs->wf[ie] = NULL;
			}
		}
	} else
#endif
	{
		orbs->data = malloc(grid2_size(grid)*atom->n_orbs*sizeof(cdouble));
		for (int ie = 0; ie < atom->n_orbs; ++ie) {
			orbs->wf[ie] = sh_wavefunc_new_from(&orbs->data[grid2_size(grid)*ie], grid, atom->m[ie]);
		}
	}
	
	return orbs;
}

void orbitals_init(orbitals_t* orbs) {
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		orbs->mpi_wf->m = orbs->atom->m[orbs->mpi_rank];
		sh_wavefunc_random_l(orbs->mpi_wf, orbs->atom->l[orbs->mpi_rank]);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			orbs->wf[ie]->m = orbs->atom->m[ie];
			sh_wavefunc_random_l(orbs->wf[ie], orbs->atom->l[ie]);
		}
	}
}

void orbitals_set_init_state(orbitals_t* orbs, cdouble* data, int l_max) {
	int const size = l_max*orbs->grid->n[iR];
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		MPI_Scatter(data, size, MPI_C_DOUBLE_COMPLEX, orbs->mpi_wf->data, size, MPI_C_DOUBLE_COMPLEX, 0, orbs->mpi_comm);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			memcpy(orbs->wf[ie]->data, &data[ie*size], size*sizeof(cdouble));
		}
	}
}

void orbitals_del(orbitals_t* orbs) {
	assert(orbs != NULL);

	for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
		if (orbs->wf[ie] != NULL) {
			sh_wavefunc_del(orbs->wf[ie]);
		}
	}
	
	free(orbs->data);
	free(orbs);
}

double orbitals_norm(orbitals_t const* orbs, sh_f mask) {
	assert(orbs != NULL);

	double res = 0.0;

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		double local_res = sh_wavefunc_norm(orbs->mpi_wf, mask)*orbs->atom->n_e[orbs->mpi_rank];
		MPI_Reduce(&local_res, &res, 1, MPI_DOUBLE, MPI_SUM, 0, orbs->mpi_comm);
	} else
#endif
	{
#pragma omp parallel for reduction(+:res)
		for (int ie=0; ie<orbs->atom->n_orbs; ++ie) {
			res += sh_wavefunc_norm(orbs->wf[ie], mask)*orbs->atom->n_e[ie];
		}
	}

	return res;
}

void orbitals_norm_ne(orbitals_t const* orbs, double n[orbs->atom->n_orbs], sh_f mask) {
	assert(orbs != NULL);

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		double n_local = sh_wavefunc_norm(orbs->mpi_wf, mask);
		MPI_Gather(&n_local, 1, MPI_DOUBLE, n, 1, MPI_DOUBLE, 0, orbs->mpi_comm);
	} else
#endif
	{
		for (int ie=0; ie<orbs->atom->n_orbs; ++ie) {
			n[ie] = sh_wavefunc_norm(orbs->wf[ie], mask);
		}
	}
}

void orbitals_normalize(orbitals_t* orbs) {
	assert(orbs != NULL);

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		sh_wavefunc_normalize(orbs->mpi_wf);
	} else
#endif
	{
#pragma omp parallel for
		for (int ie=0; ie<orbs->atom->n_orbs; ++ie) {
			sh_wavefunc_normalize(orbs->wf[ie]);
		}
	}
}

double orbitals_cos(orbitals_t const* orbs, sh_f U) {
	assert(orbs != NULL);

	double res = 0.0;

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		double res_local = sh_wavefunc_cos(orbs->mpi_wf, U)*orbs->atom->n_e[orbs->mpi_rank];
		MPI_Reduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, orbs->mpi_comm);
	} else
#endif
	{
		for (int ie=0; ie<orbs->atom->n_orbs; ++ie) {
			res += sh_wavefunc_cos(orbs->wf[ie], U)*orbs->atom->n_e[ie];
		}
	}

	return res;
}

void orbitals_n_sp(orbitals_t const* orbs, sp_grid_t const* grid, double n[grid->n[iR]*grid->n[iC]], double n_tmp[grid->n[iR]*grid->n[iC]], ylm_cache_t const* ylm_cache) {
	assert(orbs != NULL && grid != NULL && n != NULL && ylm_cache != NULL);
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
#pragma omp parallel for collapse(2)
		for (int ic = 0; ic < grid->n[iC]; ++ic) {
			for (int ir = 0; ir < grid->n[iR]; ++ir) {
				cdouble const psi = swf_get_sp(orbs->mpi_wf, grid, (int[3]){ir, ic, 0}, ylm_cache);
				n_tmp[ir + ic*grid->n[iR]] = (pow(creal(psi), 2) + pow(cimag(psi), 2))*orbs->atom->n_e[orbs->mpi_rank];
			}
		}

		MPI_Reduce(n_tmp, n, grid->n[iR]*grid->n[iC], MPI_DOUBLE, MPI_SUM, 0, orbs->mpi_comm);
	} else
#endif
	{
#pragma omp parallel
		{
#pragma omp for collapse(2)
			for (int ic = 0; ic < grid->n[iC]; ++ic) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					n[ir + ic*grid->n[iR]] = 0.0;
				}
			}

#pragma omp for collapse(2)
			for (int ic = 0; ic < grid->n[iC]; ++ic) {
				for (int ir = 0; ir < grid->n[iR]; ++ir) {
					for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
						cdouble const psi = swf_get_sp(orbs->wf[ie], grid, (int[3]){ir, ic, 0}, ylm_cache);
						n[ir + ic*grid->n[iR]] += (pow(creal(psi), 2) + pow(cimag(psi), 2))*orbs->atom->n_e[ie];
					}
				}
			}
		}
	}
}

double orbitals_n(orbitals_t const* orbs, sp_grid_t const* grid, int i[2], ylm_cache_t const* ylm_cache) {
	assert(orbs != NULL);
#ifdef _MPI
	assert(orbs->mpi_comm == MPI_COMM_NULL);
#endif

	double res = 0.0;

	for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
		cdouble const psi = swf_get_sp(orbs->wf[ie], grid, (int[3]){i[0], i[1], 0}, ylm_cache);
		res += (pow(creal(psi), 2) + pow(cimag(psi), 2))*orbs->atom->n_e[ie];
	}

	return res;
}
