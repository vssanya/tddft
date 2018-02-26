#include "orbs_spin.h"

workspace::OrbsSpin::OrbsSpin(
        const AtomCache* atom_cache,
        const ShGrid *sh_grid,
        const SpGrid *sp_grid,
        const uabs_sh_t *uabs,
        const ylm_cache_t *ylm_cache,
        int Uh_lmax,
        int Uxc_lmax,
        potential_xc_f Uxc,
        int num_threads): workspace::orbs(atom_cache, sh_grid, sp_grid, uabs, ylm_cache, Uh_lmax, Uxc_lmax, Uxc, num_threads) {
}

void workspace::OrbsSpin::init() {
    Utmp = new double[sh_grid->n[iR]]();
    Utmp_local = new double[sh_grid->n[iR]]();

    uh_tmp = new double[sh_grid->n[iR]]();

    Uee = new double[3*sh_grid->n[iR]]();

    n_sp = new double[sp_grid->n[iR]*sp_grid->n[iC]];
    n_sp_local = new double[sp_grid->n[iR]*sp_grid->n[iC]];
}


void workspace::OrbsSpin::calc_Uee(Orbitals const* orbs, int Uxc_lmax, int Uh_lmax) {
#ifdef _MPI
	if (orbs->mpi_comm == MPI_COMM_NULL || orbs->mpi_rank == 0)
#endif
	{
#pragma omp parallel for collapse(2)
		for (int il=0; il<lmax; ++il) {
			for (int ir=0; ir<sh_grid->n[iR]; ++ir) {
				Uee[ir + il*sh_grid->n[iR]] = 0.0;
			}
		}
	}

	for (int il=0; il<Uxc_lmax; ++il) {
		uxc_calc_l0(Uxc, il, orbs, Utmp, sp_grid, n_sp, n_sp_local, ylm_cache);

#ifdef _MPI
		if (orbs->mpi_comm == MPI_COMM_NULL || orbs->mpi_rank == 0)
#endif
		{
#pragma omp parallel for
			for (int ir=0; ir<sh_grid->n[iR]; ++ir) {
				Uee[ir + il*sh_grid->n[iR]] += Utmp[ir]*UXC_NORM_L[il];
			}
		}
	}

	for (int il=0; il<Uh_lmax; ++il) {
		hartree_potential(orbs, il, Utmp, Utmp_local, uh_tmp, 3);

#ifdef _MPI
		if (orbs->mpi_comm == MPI_COMM_NULL || orbs->mpi_rank == 0)
#endif
		{
#pragma omp parallel for
			for (int ir=0; ir<sh_grid->n[iR]; ++ir) {
				Uee[ir + il*sh_grid->n[iR]] += Utmp[ir];
			}
		}
	}

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		MPI_Bcast(Uee, orbs->grid->n[iR]*lmax, MPI_DOUBLE, 0, orbs->mpi_comm);
	}
#endif
}

