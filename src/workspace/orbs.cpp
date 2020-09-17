#include "orbs.h"

#include <stdlib.h>
#include <algorithm>

#include "common_alg.h"


template<typename Grid>
workspace::OrbitalsWS<Grid>::OrbitalsWS(
		Grid      const& sh_grid,
		SpGrid    const& sp_grid,
		AtomCache<Grid> const& atom_cache,
		UabsCache const& uabs,
		YlmCache  const& ylm_cache,
		int Uh_lmax,
		int Uxc_lmax,
		potential_xc_f Uxc,
		PropAtType propAtType,
		Gauge gauge,
		int num_threads
		):
    wf_ws(sh_grid, atom_cache, uabs, propAtType, gauge, num_threads),
	Uh_lmax(Uh_lmax),
	Uxc(Uxc),
	Uxc_lmax(Uxc_lmax),
	sh_grid(sh_grid),
	sp_grid(sp_grid),
	uee_grid(Grid2d(sh_grid.n[iR], 3)),
	ylm_cache(ylm_cache),
	timeApproxUeeType(TimeApproxUeeType::SIMPLE),
	gauge(gauge),
	tmpOrb(nullptr),
	tmpUee(nullptr),
	Uee(nullptr)
{	
	lmax = std::max(Uh_lmax, Uxc_lmax);
	lmax = std::max(lmax, 2);

	init();
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::init() {
	Uee = new Array2D<double>(uee_grid);
	Uee->set(0.0);

    Utmp = new double[sh_grid.n[iR]]();
    Utmp_local = new double[sh_grid.n[iR]]();

    uh_tmp = new double[sh_grid.n[iR]]();

    n_sp = new double[sp_grid.n[iR]*sp_grid.n[iT]]();
    n_sp_local = new double[sp_grid.n[iR]*sp_grid.n[iT]]();
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::setTimeApproxUeeTwoPointFor(Orbitals<Grid> const& orbs) {
	timeApproxUeeType = TimeApproxUeeType::TWO_POINT;

	if (tmpOrb != nullptr) {
		delete tmpOrb;
	}

	if (tmpUee == nullptr) {
		tmpUee = new Array2D<double>(uee_grid);
		tmpUee->set(0.0);
	}

	tmpOrb = orbs.copy();

	calc_Uee(orbs, Uxc_lmax, Uh_lmax);
}

template<typename Grid>
workspace::OrbitalsWS<Grid>::~OrbitalsWS() {
	delete[] n_sp_local;
	delete[] n_sp;
	delete[] uh_tmp;
	delete[] Utmp;
	delete[] Utmp_local;

	if (tmpOrb != nullptr) {
		delete tmpOrb;
	}

	if (tmpUee != nullptr) {
		delete tmpUee;
	}
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::calc_Uee(
		Orbitals<Grid> const& orbs,
		int Uxc_lmax,
		int Uh_lmax,
		Array2D<double>* Uee,
		std::optional<Range> rRange
		) {
	const int Nr = sh_grid.n[iR];

	if (Uee == nullptr) {
		Uee = this->Uee;
	}

#ifdef _MPI
	if (orbs.mpi_comm == MPI_COMM_NULL || orbs.mpi_rank == 0)
#endif
	{
#pragma omp parallel for collapse(2)
		for (int il=0; il<lmax; ++il) {
			for (int ir=0; ir<Nr; ++ir) {
				Uee->data[ir + il*Nr] = 0.0;
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
			for (int ir=0; ir<Nr; ++ir) {
				Uee->data[ir + il*Nr] += Utmp[ir]*UXC_NORM_L[il];
			}
		}
	}

	for (int il=0; il<Uh_lmax; ++il) {
		HartreePotential<Grid>::calc(&orbs, il, Utmp, Utmp_local, uh_tmp, 3, rRange);

#ifdef _MPI
		if (orbs.mpi_comm == MPI_COMM_NULL || orbs.mpi_rank == 0)
#endif
		{
#pragma omp parallel for
			for (int ir=0; ir<sh_grid.n[iR]; ++ir) {
				Uee->data[ir + il*Nr] += Utmp[ir];
			}
		}
	}

#ifdef _MPI
	if (orbs.mpi_comm != MPI_COMM_NULL) {
		MPI_Bcast(Uee->data, orbs.grid.n[iR]*lmax, MPI_DOUBLE, 0, orbs.mpi_comm);
	}
#endif
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop_simple(Orbitals<Grid>& orbs, field_t const* field, double t, double dt, bool calc_uee, bool* activeOrbs, int const* dt_count) {
	double Et = field_E(field, t + dt/2);
	double At = -field_A(field, t + dt/2);

	if (gauge == Gauge::VELOCITY) {
		Et = 0;
	}

	if (calc_uee) {
		calc_Uee(orbs, Uxc_lmax, Uh_lmax);
	}

	typename workspace::WavefuncWS<Grid>::sh_f Ul[3] = {
        [this](int ir, int l, int m) -> double {
			double const r = sh_grid.r(ir);
            return l*(l+1)/(2*r*r) + wf_ws.atom_cache.u(ir) + Uee->data[ir + 0*sh_grid.n[iR]] + plm(l,m)*Uee->data[ir + 2*sh_grid.n[iR]];
		},
        [Et, this](int ir, int l, int m) -> double {
			double const r = sh_grid.r(ir);
			return clm(l, m)*(r*Et + Uee->data[ir + 1*sh_grid.n[iR]]);
		},
        [this](int ir, int l, int m) -> double {
			return qlm(l, m)*Uee->data[ir + 2*sh_grid.n[iR]];
		}
	};

	sh_f Al[2] = {
		[At](int ir, int l, int m) -> double {
			return At*clm(l,m);
		},
		[this, At](int ir, int l, int m) -> double {
			double const r = sh_grid.r(ir);
			return At*(l+1)*clm(l,m)/r;
		}
	};

	for (int ie = 0; ie < orbs.atom.countOrbs; ++ie) {
		if (orbs.wf[ie] != nullptr && (activeOrbs == nullptr || activeOrbs[ie])) {
			auto count = (dt_count == nullptr) ? 1 : dt_count[ie];
			for (int i = 0; i < count; ++i) {
				switch (gauge) {
					case Gauge::LENGTH:
						wf_ws.prop_common(*orbs.wf[ie], dt/count, lmax, Ul);
						break;
					case Gauge::VELOCITY:
						wf_ws.prop_common(*orbs.wf[ie], dt/count, lmax, Ul, Al);
						break;
				}
				wf_ws.prop_abs(*orbs.wf[ie], dt/count);
			}
		}
	}
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop_two_point(Orbitals<Grid>& orbs, field_t const* field, double t, double dt, bool calc_uee, bool* activeOrbs, int const* dt_count) {
	if (calc_uee) {
		// Calc Uee(t)
		calc_Uee(orbs, Uxc_lmax, Uh_lmax, tmpUee);

		// Calc Uee(t+dt/2)
#pragma omp parallel for collapse(2)
		for (int l=0; l<3; l++) {
			for (int ir=0; ir<Uee->grid.n[iR]; ir++) {
				(*Uee)(ir, l) = 2*(*tmpUee)(ir, l) - (*Uee)(ir, l);
			}
		}

		double err = 1.0;
		int max_iterations = 2;
		int n = 0;
		
		while (err >= 1.0 and n < max_iterations) {
			// Calc orb(t+dt)
			orbs.copy(*tmpOrb);
			prop_simple(*tmpOrb, field, t, dt, false, activeOrbs, dt_count);

			// calc orb(t+dt/2)
			tmpOrb->mean(orbs);

			Uee->copy(tmpUee);
			// Calc Uee(t+dt/2)
			calc_Uee(*tmpOrb, Uxc_lmax, Uh_lmax);

			err = 0.0;
#pragma omp parallel for collapse(2) reduction(+:err)
			for (int l=0; l<3; l++) {
				for (int ir=0; ir<Uee->grid.n[iR]; ir++) {
					err += abs((*tmpUee)(ir, l) - (*Uee)(ir, l));
				}
			}

			n++;
		}
	}

	prop_simple(orbs, field, t, dt, false, activeOrbs, dt_count);
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop(Orbitals<Grid>& orbs, field_t const* field, double t, double dt, bool calc_uee, bool* activeOrbs, int const* dt_count) {
	if (timeApproxUeeType == TimeApproxUeeType::SIMPLE) {
		prop_simple(orbs, field, t, dt, calc_uee, activeOrbs, dt_count);
	} else {
		prop_two_point(orbs, field, t, dt, calc_uee, activeOrbs, dt_count);
	}
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop_img(Orbitals<Grid>& orbs, double dt, bool activeOrbs[], int const dt_count[], bool calc_uee) {
    auto lmax = std::max(std::max(1, Uxc_lmax), Uh_lmax);
	if (calc_uee) {
		calc_Uee(orbs, Uxc_lmax, Uh_lmax);
	}

	typename workspace::WavefuncWS<Grid>::sh_f Ul[3] = {
        [this](int ir, int l, int m) -> double {
			double const r = sh_grid.r(ir);
            return l*(l+1)/(2*r*r) + wf_ws.atom_cache.u(ir) + Uee->data[ir + 0*sh_grid.n[iR]];
        },
        [this](int ir, int l, int m) -> double {
            return clm(l, m)*Uee->data[ir + 1*sh_grid.n[iR]];
        },
        [this](int ir, int l, int m) -> double {
            return qlm(l, m)*Uee->data[ir + 2*sh_grid.n[iR]];
        }
    };

    for (int ie = 0; ie < orbs.atom.countOrbs; ++ie) {
        if (orbs.wf[ie] != nullptr && (activeOrbs == nullptr || activeOrbs[ie])) {
			auto count = (dt_count == nullptr) ? 1 : dt_count[ie];
			for (int i = 0; i < count; ++i) {
				wf_ws.prop_common(*orbs.wf[ie], -I*dt/count, lmax, Ul);
			}
        }
    }
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop_ha(Orbitals<Grid>& orbs, double dt,
		bool calc_uee,
		bool activeOrbs[]
) {
    auto lmax = std::max(std::max(1, Uxc_lmax), Uh_lmax);
	if (calc_uee) {
		calc_Uee(orbs, Uxc_lmax, Uh_lmax);
	}

	typename workspace::WavefuncWS<Grid>::sh_f Ul[3] = {
        [this](int ir, int l, int m) -> double {
			double const r = sh_grid.r(ir);
            return l*(l+1)/(2*r*r) + wf_ws.atom_cache.u(ir) + Uee->data[ir + 0*sh_grid.n[iR]];
        },
        [this](int ir, int l, int m) -> double {
            return clm(l, m)*Uee->data[ir + 1*sh_grid.n[iR]];
        },
        [this](int ir, int l, int m) -> double {
            return qlm(l, m)*Uee->data[ir + 2*sh_grid.n[iR]];
        }
    };

    for (int ie = 0; ie < orbs.atom.countOrbs; ++ie) {
        if (orbs.wf[ie] != nullptr && (activeOrbs == nullptr || activeOrbs[ie])) {
            wf_ws.prop_common(*orbs.wf[ie], dt, lmax, Ul);
        }
    }
}

template<typename Grid>
void workspace::OrbitalsWS<Grid>::prop_abs(Orbitals<Grid>& orbs, double dt, bool activeOrbs[]) {
	for (int ie = 0; ie < orbs.atom.countOrbs; ++ie) {
		if (orbs.wf[ie] != nullptr && (activeOrbs == nullptr || activeOrbs[ie])) {
			wf_ws.prop_abs(*orbs.wf[ie], dt);
		}
	}
}

template class workspace::OrbitalsWS<ShGrid>;
template class workspace::OrbitalsWS<ShNotEqudistantGrid>;
