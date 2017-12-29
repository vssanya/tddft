#include "orbs.h"

#include <stdlib.h>
#include <algorithm>

#include "common_alg.h"


workspace::orbs::orbs(sh_grid_t const* sh_grid, sp_grid_t const* sp_grid, uabs_sh_t const* uabs, ylm_cache_t const* ylm_cache, int Uh_lmax, int Uxc_lmax, potential_xc_f Uxc, int num_threads):
	wf_ws(sh_grid, uabs, num_threads),
	Uh_lmax(Uh_lmax),
	Uxc(Uxc),
	Uxc_lmax(Uxc_lmax),
	sh_grid(sh_grid),
	sp_grid(sp_grid),
	ylm_cache(ylm_cache)
{	
	lmax = std::max(Uh_lmax, Uxc_lmax);
	lmax = std::max(lmax, 2);

	Utmp = new double[sh_grid->n[iR]];
	Utmp_local = new double[sh_grid->n[iR]];

	uh_tmp = new double[sh_grid->n[iR]];

	Uee = new double[3*sh_grid->n[iR]]();

	n_sp = new double[sp_grid->n[iR]*sp_grid->n[iC]];
	n_sp_local = new double[sp_grid->n[iR]*sp_grid->n[iC]];
}

workspace::orbs::~orbs() {
	delete[] n_sp_local;
	delete[] n_sp;
	delete[] uh_tmp;
	delete[] Utmp;
	delete[] Utmp_local;
	delete[] Uee;
}

void workspace::orbs::calc_Uee(orbitals_t const* orbs, int Uxc_lmax, int Uh_lmax) {
	double const UXC_NORM_L[] = {sqrt(1.0/(4.0*M_PI)), sqrt(3.0/(4*M_PI)), sqrt(5.0/(4*M_PI))};

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

void workspace::orbs::prop(orbitals_t* orbs, atom_t const* atom, field_t const* field, double t, double dt, bool calc_uee) {
	double Et = field_E(field, t + dt/2);

	if (calc_uee) {
		calc_Uee(orbs, Uxc_lmax, Uh_lmax);
	}

	sh_f Ul[3] = {
		[atom, this](sh_grid_t const* grid, int ir, int l, int m) -> double {
			double const r = sh_grid_r(grid, ir);
			return l*(l+1)/(2*r*r) + atom->u(atom, grid, ir) + Uee[ir] + plm(l,m)*Uee[ir + 2*grid->n[iR]];
		},
		[Et, this](sh_grid_t const* grid, int ir, int l, int m) -> double {
			double const r = sh_grid_r(grid, ir);
			return clm(l, m)*(r*Et + Uee[ir + grid->n[iR]]);
		},
		[this](sh_grid_t const* grid, int ir, int l, int m) -> double {
			return qlm(l, m)*Uee[ir + 2*grid->n[iR]];
		}
	};

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		wf_ws.prop_common(*orbs->mpi_wf, dt, lmax, Ul, atom->Z, atom->u_type);
		wf_ws.prop_abs(*orbs->mpi_wf, dt);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			wf_ws.prop_common(*orbs->wf[ie], dt, lmax, Ul, atom->Z, atom->u_type);
			wf_ws.prop_abs(*orbs->wf[ie], dt);
		}
	}
}

void workspace::orbs::prop_img(orbitals_t* orbs, atom_t const* atom, double dt) {
	calc_Uee(orbs, std::min(1, Uxc_lmax), std::min(1, Uh_lmax));

	sh_f Ul[1] = {
		[atom, this](sh_grid_t const* grid, int ir, int l, int m) -> double {
			double const r = sh_grid_r(grid, ir);
			return l*(l+1)/(2*r*r) + atom->u(atom, grid, ir) + Uee[ir];
		}
	};

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		wf_ws.prop_common(*orbs->mpi_wf, -I*dt, 1, Ul, atom->Z, atom->u_type);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			wf_ws.prop_common(*orbs->wf[ie], -I*dt, 1, Ul, atom->Z, atom->u_type);
		}
	}
}
