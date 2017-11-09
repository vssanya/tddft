#include "orbs.h"

#include <stdlib.h>

#include "common_alg.h"


ws_orbs_t* ws_orbs_alloc(
		sh_grid_t const* sh_grid,
		sp_grid_t const* sp_grid,
		uabs_sh_t const* uabs,
		ylm_cache_t const* ylm_cache,
		int Uh_lmax,
		int Uxc_lmax,
		potential_xc_f Uxc,
		int num_threads
		) {	
	ws_orbs_t* ws = malloc(sizeof(ws_orbs_t));

	ws->sh_grid = sh_grid;
	ws->sp_grid = sp_grid;
	ws->ylm_cache = ylm_cache;

	ws->wf_ws = ws_wf_new(sh_grid, uabs, num_threads);

	ws->Uh_lmax = Uh_lmax;
	ws->Uxc_lmax = Uxc_lmax;

	ws->lmax = MAX(Uh_lmax, Uxc_lmax);
	ws->lmax = MAX(ws->lmax, 2);

	ws->Uxc = Uxc;

	ws->Utmp = malloc(sizeof(double)*sh_grid->n[iR]);
	ws->Utmp_local = malloc(sizeof(double)*sh_grid->n[iR]);

	ws->uh_tmp = malloc(sizeof(double)*sh_grid->n[iR]);

	ws->Uee = calloc(3*sh_grid->n[iR], sizeof(double));

	ws->n_sp = malloc(sizeof(double)*ws->sp_grid->n[iR]*ws->sp_grid->n[iC]);
	ws->n_sp_local = malloc(sizeof(double)*ws->sp_grid->n[iR]*ws->sp_grid->n[iC]);

	return ws;
}

void ws_orbs_free(ws_orbs_t* ws) {
	free(ws->n_sp_local);
	free(ws->n_sp);
	free(ws->uh_tmp);

	free(ws->Utmp);
	free(ws->Utmp_local);

	free(ws->Uee);

	ws_wf_del(ws->wf_ws);

	free(ws);
}

void ws_orbs_calc_Uee(ws_orbs_t* ws,
		orbitals_t const* orbs,
		int Uxc_lmax,
		int Uh_lmax) {
	double const UXC_NORM_L[] = {sqrt(1.0/(4.0*M_PI)), sqrt(3.0/(4*M_PI)), sqrt(5.0/(4*M_PI))};

#ifdef _MPI
	if (orbs->mpi_comm == MPI_COMM_NULL || orbs->mpi_rank == 0)
#endif
	{
#pragma omp parallel for collapse(2)
		for (int il=0; il<ws->lmax; ++il) {
			for (int ir=0; ir<ws->sh_grid->n[iR]; ++ir) {
				ws->Uee[ir + il*ws->sh_grid->n[iR]] = 0.0;
			}
		}
	}

	for (int il=0; il<Uxc_lmax; ++il) {
		uxc_calc_l0(ws->Uxc, il, orbs, ws->Utmp, ws->sp_grid, ws->n_sp, ws->n_sp_local, ws->ylm_cache);

#ifdef _MPI
		if (orbs->mpi_comm == MPI_COMM_NULL || orbs->mpi_rank == 0)
#endif
		{
#pragma omp parallel for
			for (int ir=0; ir<ws->sh_grid->n[iR]; ++ir) {
				ws->Uee[ir + il*ws->sh_grid->n[iR]] += ws->Utmp[ir]*UXC_NORM_L[il];
			}
		}
	}

	for (int il=0; il<Uh_lmax; ++il) {
		hartree_potential(orbs, il, ws->Utmp, ws->Utmp_local, ws->uh_tmp, 3);

#ifdef _MPI
		if (orbs->mpi_comm == MPI_COMM_NULL || orbs->mpi_rank == 0)
#endif
		{
#pragma omp parallel for
			for (int ir=0; ir<ws->sh_grid->n[iR]; ++ir) {
				ws->Uee[ir + il*ws->sh_grid->n[iR]] += ws->Utmp[ir];
			}
		}
	}

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		MPI_Bcast(ws->Uee, orbs->grid->n[iR]*ws->lmax, MPI_DOUBLE, 0, orbs->mpi_comm);
	}
#endif

}

void ws_orbs_prop(
		ws_orbs_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		field_t const* field,
		double t,
		double dt,
		bool calc_uee
		) {
	double Et = field_E(field, t + dt/2);

	if (calc_uee) {
		ws_orbs_calc_Uee(ws, orbs, ws->Uxc_lmax, ws->Uh_lmax);
	}

	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(atom, grid, ir) + ws->Uee[ir] + plm(l,m)*ws->Uee[ir + 2*grid->n[iR]];
	}

	double Ul1(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return clm(l, m)*(r*Et + ws->Uee[ir + grid->n[iR]]);
	}

	double Ul2(sh_grid_t const* grid, int ir, int l, int m) {
		return qlm(l, m)*ws->Uee[ir + 2*grid->n[iR]];
	}

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		ws_wf_prop_common(ws->wf_ws, orbs->mpi_wf, dt, ws->lmax, (sh_f[3]){Ul0, Ul1, Ul2}, ws->wf_ws->uabs, atom->Z, atom->u_type);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			ws_wf_prop_common(ws->wf_ws, orbs->wf[ie], dt, ws->lmax, (sh_f[3]){Ul0, Ul1, Ul2}, ws->wf_ws->uabs, atom->Z, atom->u_type);
		}
	}
}

void ws_orbs_prop_img(
		ws_orbs_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		double dt
		) {
	ws_orbs_calc_Uee(ws, orbs, MIN(1, ws->Uxc_lmax), MIN(1, ws->Uh_lmax));

	double Ul0(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(atom, grid, ir) + ws->Uee[ir];
	}

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		ws_wf_prop_common(ws->wf_ws, orbs->mpi_wf, -I*dt, 1, (sh_f[1]){Ul0}, &uabs_zero, atom->Z, atom->u_type);
	} else
#endif
	{
		for (int ie = 0; ie < orbs->atom->n_orbs; ++ie) {
			ws_wf_prop_common(ws->wf_ws, orbs->wf[ie], -I*dt, 1, (sh_f[1]){Ul0}, &uabs_zero, atom->Z, atom->u_type);
		}
	}
}
