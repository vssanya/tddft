#include "orbitals.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>


orbitals_t* orbials_new(int ne, sh_grid_t const* grid, MPI_Comm mpi_comm) {
	orbitals_t* orbs = malloc(sizeof(orbitals_t));

	orbs->ne = ne;
	orbs->grid = grid;

	orbs->wf = malloc(sizeof(sh_wavefunc_t*)*ne);

#ifdef _MPI
	orbs->mpi_comm = mpi_comm;

	if (mpi_comm != MPI_COMM_NULL) {
		MPI_Comm_rank(orbs->mpi_comm, &orbs->mpi_rank);

		orbs->data = malloc(grid2_size(grid)*sizeof(cdouble));
		for (int ie = 0; ie < ne; ++ie) {
			if (ie == orbs->mpi_rank) {
				orbs->wf[ie] = sh_wavefunc_new_from(orbs->data, grid, 0);
				orbs->mpi_wf = orbs->wf[ie];
			} else {
				orbs->wf[ie] = NULL;
			}
		}
	} else
#endif
	{
		orbs->data = malloc(grid2_size(grid)*ne*sizeof(cdouble));
		for (int ie = 0; ie < ne; ++ie) {
			orbs->wf[ie] = sh_wavefunc_new_from(&orbs->data[grid2_size(grid)*ie], grid, 0);
		}
	}
	
	return orbs;
}

void orbitals_set_init_state(orbitals_t* orbs, cdouble* data, int l_max) {
	int const size = l_max*orbs->grid->n[iR];
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		MPI_Scatter(data, size, MPI_C_DOUBLE_COMPLEX, orbs->mpi_wf->data, size, MPI_C_DOUBLE_COMPLEX, 0, orbs->mpi_comm);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->ne; ++ie) {
			memcpy(&data[ie*size], orbs->wf[ie]->data, size*sizeof(cdouble));
		}
	}
}

void orbitals_del(orbitals_t* orbs) {
	for (int ie = 0; ie < orbs->ne; ++ie) {
		if (orbs->wf[ie] != NULL) {
			sh_wavefunc_del(orbs->wf[ie]);
		}
	}
	
	free(orbs->data);
	free(orbs);
}

double orbitals_n(orbitals_t const* orbs, sp_grid_t const* grid, int i[2], ylm_cache_t const* ylm_cache) {
#ifdef _MPI
	assert(orbs->mpi_comm == MPI_COMM_NULL);
#endif

	double res = 0.0;

	for (int ie = 0; ie < orbs->ne; ++ie) {
		cdouble const psi = swf_get_sp(orbs->wf[ie], grid, (int[3]){i[0], i[1], 0}, ylm_cache);
		res += pow(creal(psi), 2) + pow(cimag(psi), 2);
	}

	return 2.0*res;
}

double orbitals_norm(orbitals_t const* orbs) {
	double res = 0.0;

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		res = sh_wavefunc_norm(orbs->mpi_wf);
		MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, orbs->mpi_comm);
	} else
#endif
	{
#pragma omp parallel for reduction(+:res)
		for (int ie=0; ie<orbs->ne; ++ie) {
			res += sh_wavefunc_norm(orbs->wf[ie]);
		}
	}

	return 2*res;
}

void orbitals_normalize(orbitals_t* orbs) {
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		sh_wavefunc_normalize(orbs->mpi_wf);
	} else
#endif
	{
#pragma omp parallel for
		for (int ie=0; ie<orbs->ne; ++ie) {
			sh_wavefunc_normalize(orbs->wf[ie]);
		}
	}
}

double orbitals_cos(orbitals_t const* orbs, sh_f U) {
	double res = 0.0;

#ifdef _MPI
	if (orbs->mpi_comm == MPI_COMM_NULL) {
		res = sh_wavefunc_cos(orbs->mpi_wf, U);
		MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, orbs->mpi_comm);
	} else
#endif
	{
#pragma omp parallel for reduction(+:res)
		for (int ie=0; ie<orbs->ne; ++ie) {
			res += sh_wavefunc_cos(orbs->wf[ie], U);
		}
	}

	return 2*res;
}

void orbitals_n_sp(orbitals_t const* orbs, sp_grid_t const* grid, double n[grid->n[iR]*grid->n[iC]], double n_tmp[grid->n[iR]*grid->n[iC]], ylm_cache_t const* ylm_cache) {
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		sh_wavefunc_n_sp(orbs->mpi_wf, grid, n_tmp, ylm_cache);
		n[0] = 2;
		MPI_Reduce(n_tmp, n, grid->n[iR]*grid->n[iC], MPI_DOUBLE, MPI_SUM, MPI_ROOT, orbs->mpi_comm);
	} else
#endif
	{
		assert(false);
	}
}
