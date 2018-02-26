#include "orbs.h"

#include <stdlib.h>
#include <algorithm>

#include "common_alg.h"


workspace::orbs::orbs(const AtomCache* atom_cache, ShGrid const* sh_grid, SpGrid const* sp_grid, uabs_sh_t const* uabs, ylm_cache_t const* ylm_cache, int Uh_lmax, int Uxc_lmax, potential_xc_f Uxc, int num_threads):
    wf_ws(atom_cache, sh_grid, uabs, num_threads),
	Uh_lmax(Uh_lmax),
	Uxc(Uxc),
	Uxc_lmax(Uxc_lmax),
	sh_grid(sh_grid),
	sp_grid(sp_grid),
	ylm_cache(ylm_cache)
{	
	lmax = std::max(Uh_lmax, Uxc_lmax);
	lmax = std::max(lmax, 2);

	init();
}

void workspace::orbs::init() {
    Utmp = new double[sh_grid->n[iR]]();
    Utmp_local = new double[sh_grid->n[iR]]();

    uh_tmp = new double[sh_grid->n[iR]]();

	Uee = new double[3*sh_grid->n[iR]]();

    n_sp = new double[sp_grid->n[iR]*sp_grid->n[iC]]();
    n_sp_local = new double[sp_grid->n[iR]*sp_grid->n[iC]]();
}

workspace::orbs::~orbs() {
	delete[] n_sp_local;
	delete[] n_sp;
	delete[] uh_tmp;
	delete[] Utmp;
	delete[] Utmp_local;
	delete[] Uee;
}

void workspace::orbs::calc_Uee(Orbitals const* orbs, int Uxc_lmax, int Uh_lmax) {
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

void workspace::orbs::prop(Orbitals* orbs, field_t const* field, double t, double dt, bool calc_uee) {
	double Et = field_E(field, t + dt/2);

	if (calc_uee) {
		calc_Uee(orbs, Uxc_lmax, Uh_lmax);
	}

	sh_f Ul[3] = {
        [this](ShGrid const* grid, int ir, int l, int m) -> double {
			double const r = grid->r(ir);
            return l*(l+1)/(2*r*r) + wf_ws.atom_cache->u(ir) + Uee[ir] + plm(l,m)*Uee[ir + 2*grid->n[iR]];
		},
        [Et, this](ShGrid const* grid, int ir, int l, int m) -> double {
			double const r = grid->r(ir);
			return clm(l, m)*(r*Et + Uee[ir + grid->n[iR]]);
		},
        [this](ShGrid const* grid, int ir, int l, int m) -> double {
			return qlm(l, m)*Uee[ir + 2*grid->n[iR]];
		}
	};

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
        wf_ws.prop_common(*orbs->mpi_wf, dt, lmax, Ul);
		wf_ws.prop_abs(*orbs->mpi_wf, dt);
	} else
#endif
	{
        for (int ie = 0; ie < orbs->atom.countOrbs; ++ie) {
            wf_ws.prop_common(*orbs->wf[ie], dt, lmax, Ul);
			wf_ws.prop_abs(*orbs->wf[ie], dt);
		}
	}
}

void workspace::orbs::prop_img(Orbitals* orbs, double dt) {
    auto lmax = std::max(std::max(1, Uxc_lmax), Uh_lmax);
    calc_Uee(orbs, Uxc_lmax, Uh_lmax);

    sh_f Ul[3] = {
        [this](ShGrid const* grid, int ir, int l, int m) -> double {
			double const r = grid->r(ir);
            return l*(l+1)/(2*r*r) + wf_ws.atom_cache->u(ir) + Uee[ir];
        },
        [this](ShGrid const* grid, int ir, int l, int m) -> double {
            return clm(l, m)*Uee[ir + grid->n[iR]];
        },
        [this](ShGrid const* grid, int ir, int l, int m) -> double {
            return qlm(l, m)*Uee[ir + 2*grid->n[iR]];
        }
    };

    for (int ie = 0; ie < orbs->atom.countOrbs; ++ie) {
        if (orbs->wf[ie] != nullptr) {
            wf_ws.prop_common(*orbs->wf[ie], -I*dt, lmax, Ul);
        }
    }
}
