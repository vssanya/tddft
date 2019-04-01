#include "orbs.h"

#include <stdlib.h>
#include <algorithm>

#include "common_alg.h"


template<typename Grid>
workspace::OrbitalsWS<Grid>::OrbitalsWS(
		Grid      const& sh_grid,
		SpGrid    const& sp_grid,
		AtomCache const& atom_cache,
		UabsCache const& uabs,
		YlmCache  const& ylm_cache,
		int Uh_lmax,
		int Uxc_lmax,
		potential_xc_f Uxc,
		PropAtType propAtType,
		int num_threads
		):
    wf_ws(sh_grid, atom_cache, uabs, propAtType, num_threads),
	Uh_lmax(Uh_lmax),
	Uxc(Uxc),
	Uxc_lmax(Uxc_lmax),
	sh_grid(sh_grid),
	sp_grid(sp_grid),
	ylm_cache(ylm_cache),
	timeApproxUeeType(TimeApproxUeeType::SIMPLE),
	tmpOrb(nullptr),
	tmpUee(nullptr)
{	
	lmax = std::max(Uh_lmax, Uxc_lmax);
	lmax = std::max(lmax, 2);

	init();
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::init() {
    Utmp = new double[sh_grid.n[iR]]();
    Utmp_local = new double[sh_grid.n[iR]]();

    uh_tmp = new double[sh_grid.n[iR]]();

	Uee = new double[3*sh_grid.n[iR]]();

    n_sp = new double[sp_grid.n[iR]*sp_grid.n[iC]]();
    n_sp_local = new double[sp_grid.n[iR]*sp_grid.n[iC]]();
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::setTimeApproxUeeTwoPointFor(Orbitals<Grid> const& orbs) {
	timeApproxUeeType = TimeApproxUeeType::TWO_POINT;

	if (tmpOrb != nullptr) {
		delete tmpOrb;
	}

	if (tmpUee == nullptr) {
		tmpUee = new double[3*sh_grid.n[iR]]();
	}

	tmpOrb = orbs.copy();
}

template<typename Grid>
workspace::OrbitalsWS<Grid>::~OrbitalsWS() {
	delete[] n_sp_local;
	delete[] n_sp;
	delete[] uh_tmp;
	delete[] Utmp;
	delete[] Utmp_local;
	delete[] Uee;

	if (tmpOrb != nullptr) {
		delete tmpOrb;
	}

	if (tmpUee != nullptr) {
		delete[] tmpUee;
	}
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::calc_Uee(Orbitals<Grid> const& orbs, int Uxc_lmax, int Uh_lmax, double* Uee) {
	if (Uee == nullptr) {
		Uee = this->Uee;
	}

#ifdef _MPI
	if (orbs.mpi_comm == MPI_COMM_NULL || orbs.mpi_rank == 0)
#endif
	{
#pragma omp parallel for collapse(2)
		for (int il=0; il<lmax; ++il) {
			for (int ir=0; ir<sh_grid.n[iR]; ++ir) {
				Uee[ir + il*sh_grid.n[iR]] = 0.0;
			}
		}
	}

	for (int il=0; il<Uxc_lmax; ++il) {
		XCPotential<Grid>::calc_l0(Uxc, il, &orbs, Utmp, &sp_grid, n_sp, n_sp_local, &ylm_cache);

#ifdef _MPI
		if (orbs.mpi_comm == MPI_COMM_NULL || orbs.mpi_rank == 0)
#endif
		{
#pragma omp parallel for
			for (int ir=0; ir<sh_grid.n[iR]; ++ir) {
				Uee[ir + il*sh_grid.n[iR]] += Utmp[ir]*UXC_NORM_L[il];
			}
		}
	}

	for (int il=0; il<Uh_lmax; ++il) {
		HartreePotential<Grid>::calc(&orbs, il, Utmp, Utmp_local, uh_tmp, 3);

#ifdef _MPI
		if (orbs.mpi_comm == MPI_COMM_NULL || orbs.mpi_rank == 0)
#endif
		{
#pragma omp parallel for
			for (int ir=0; ir<sh_grid.n[iR]; ++ir) {
				Uee[ir + il*sh_grid.n[iR]] += Utmp[ir];
			}
		}
	}

#ifdef _MPI
	if (orbs.mpi_comm != MPI_COMM_NULL) {
		MPI_Bcast(Uee, orbs.grid.n[iR]*lmax, MPI_DOUBLE, 0, orbs.mpi_comm);
	}
#endif
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop_simple(Orbitals<Grid>& orbs, field_t const* field, double t, double dt, bool calc_uee) {
	double Et = field_E(field, t + dt/2);

	if (calc_uee) {
		calc_Uee(orbs, Uxc_lmax, Uh_lmax);
	}

	typename workspace::WavefuncWS<Grid>::sh_f Ul[3] = {
        [this](int ir, int l, int m) -> double {
			double const r = sh_grid.r(ir);
            return l*(l+1)/(2*r*r) + wf_ws.atom_cache.u(ir) + Uee[ir] + plm(l,m)*Uee[ir + 2*sh_grid.n[iR]];
		},
        [Et, this](int ir, int l, int m) -> double {
			double const r = sh_grid.r(ir);
			return clm(l, m)*(r*Et + Uee[ir + sh_grid.n[iR]]);
		},
        [this](int ir, int l, int m) -> double {
			return qlm(l, m)*Uee[ir + 2*sh_grid.n[iR]];
		}
	};

	for (int ie = 0; ie < orbs.atom.countOrbs; ++ie) {
		if (orbs.wf[ie] != nullptr) {
			wf_ws.prop_common(*orbs.wf[ie], dt, lmax, Ul);
			wf_ws.prop_abs(*orbs.wf[ie], dt);
		}
	}
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop_two_point(Orbitals<Grid>& orbs, field_t const* field, double t, double dt, bool calc_uee) {
	if (calc_uee) {
		// Calc orb(t+dt) and Uee(t)
		orbs.copy(*tmpOrb);
		prop_simple(*tmpOrb, field, t, dt, true);

		// Calc Uee(t+dt)
		calc_Uee(*tmpOrb, Uxc_lmax, Uh_lmax, tmpUee);

		int Nr = sh_grid.n[iR];

#pragma omp parallel for
		for (int ir_l=0; ir_l<Nr*3; ir_l++) {
			Uee[ir_l] = 0.5*(Uee[ir_l] + tmpUee[ir_l]);
		}
	}

	prop_simple(orbs, field, t, dt, false);
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop(Orbitals<Grid>& orbs, field_t const* field, double t, double dt, bool calc_uee) {
	if (timeApproxUeeType == TimeApproxUeeType::SIMPLE) {
		prop_simple(orbs, field, t, dt, calc_uee);
	} else {
		prop_two_point(orbs, field, t, dt, calc_uee);
	}
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop_img(Orbitals<Grid>& orbs, double dt) {
    auto lmax = std::max(std::max(1, Uxc_lmax), Uh_lmax);
    calc_Uee(orbs, Uxc_lmax, Uh_lmax);

	typename workspace::WavefuncWS<Grid>::sh_f Ul[3] = {
        [this](int ir, int l, int m) -> double {
			double const r = sh_grid.r(ir);
            return l*(l+1)/(2*r*r) + wf_ws.atom_cache.u(ir) + Uee[ir];
        },
        [this](int ir, int l, int m) -> double {
            return clm(l, m)*Uee[ir + sh_grid.n[iR]];
        },
        [this](int ir, int l, int m) -> double {
            return qlm(l, m)*Uee[ir + 2*sh_grid.n[iR]];
        }
    };

    for (int ie = 0; ie < orbs.atom.countOrbs; ++ie) {
        if (orbs.wf[ie] != nullptr) {
            wf_ws.prop_common(*orbs.wf[ie], -I*dt, lmax, Ul);
        }
    }
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop_ha(Orbitals<Grid>& orbs, double dt) {
    auto lmax = std::max(std::max(1, Uxc_lmax), Uh_lmax);
    calc_Uee(orbs, Uxc_lmax, Uh_lmax);

	typename workspace::WavefuncWS<Grid>::sh_f Ul[3] = {
        [this](int ir, int l, int m) -> double {
			double const r = sh_grid.r(ir);
            return l*(l+1)/(2*r*r) + wf_ws.atom_cache.u(ir) + Uee[ir];
        },
        [this](int ir, int l, int m) -> double {
            return clm(l, m)*Uee[ir + sh_grid.n[iR]];
        },
        [this](int ir, int l, int m) -> double {
            return qlm(l, m)*Uee[ir + 2*sh_grid.n[iR]];
        }
    };

    for (int ie = 0; ie < orbs.atom.countOrbs; ++ie) {
        if (orbs.wf[ie] != nullptr) {
            wf_ws.prop_common(*orbs.wf[ie], dt, lmax, Ul);
        }
    }
}

template class workspace::OrbitalsWS<ShGrid>;
template class workspace::OrbitalsWS<ShNotEqudistantGrid>;
