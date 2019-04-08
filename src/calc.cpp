#include "calc.h"
#include "utils.h"
#include "abs_pot.h"


template<class Grid>
double calc_wf_az_with_polarization(Wavefunc<Grid> const* wf, const AtomCache<Grid> &atom_cache, double const Upol[], double const dUpol_dr[], field_t const* field, double t) {
    auto func_cos = [&](int ir, int il, int m) -> double {
        return atom_cache.dudz(ir);
    };

    auto func_cos2 = [&](int ir, int il, int m) -> double {
        return dUpol_dr[ir];
    };

    auto func_sin2 = [&](int ir, int il, int m) -> double {
        return Upol[ir]/wf->grid.r(ir);
    };

	double E = field_E(field, t);
	return - E*(1 + wf->cos2(func_cos2) + wf->sin2(func_sin2)) - wf->cos(func_cos);
}

template<class Grid>
double calc_orbs_az(Orbitals<Grid> const& orbs, const AtomCache<Grid> &atom_cache, field_t const* field, double t) {
    auto func = [&](int ir, int il, int m) -> double {
        return atom_cache.dudz(ir);
    };
    return - field_E(field, t)*atom_cache.atom.countElectrons - orbs.cos(func);
}

template<class Grid>
void calc_orbs_az_ne(Orbitals<Grid> const* orbs, const AtomCache<Grid>& atom_cache, field_t const* field, double t, double* az) {
    auto func = [&](int ir, int il, int m) -> double {
        return atom_cache.dudz(ir);
    };
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
        double az_local = - (field_E(field, t) + orbs->mpi_wf->cos(func))*orbs->atom.orbs[orbs->mpi_rank].countElectrons;
		MPI_Gather(&az_local, 1, MPI_DOUBLE, az, 1, MPI_DOUBLE, 0, orbs->mpi_comm);
	} else
#endif
	{
		for (int ie=0; ie<orbs->atom.countOrbs; ++ie) {
            az[ie] = - (field_E(field, t) + orbs->wf[ie]->cos(func))*orbs->atom.orbs[ie].countElectrons;
		}
	}
}

template<class Grid>
void calc_orbs_az_ne_Vee_0(Orbitals<Grid> const* orbs, Array2D<double>& Uee, Array2D<double>& dUeedr, const AtomCache<Grid>& atom_cache, field_t const* field, double t, double* az) {
#pragma omp parallel for
	for (int ir=0; ir<orbs->grid.n[iR]; ir++) {
		dUeedr(ir, 0) = orbs->grid.d_dr(&Uee(0, 0), ir);
	}

    auto func = [&](int ir, int il, int m) -> double {
        return atom_cache.dudz(ir) + dUeedr(ir, 0);
    };

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
        double az_local = - (field_E(field, t) + orbs->mpi_wf->cos(func))*orbs->atom.orbs[orbs->mpi_rank].countElectrons;
		MPI_Gather(&az_local, 1, MPI_DOUBLE, az, 1, MPI_DOUBLE, 0, orbs->mpi_comm);
	} else
#endif
	{
		for (int ie=0; ie<orbs->atom.countOrbs; ++ie) {
            az[ie] = - (field_E(field, t) + orbs->wf[ie]->cos(func))*orbs->atom.orbs[ie].countElectrons;
		}
	}
}

template<class Grid>
void calc_orbs_az_ne_Vee_1(Orbitals<Grid> const* orbs, Array2D<double>& Uee, Array2D<double>& dUeedr, const AtomCache<Grid>& atom_cache, field_t const* field, double t, double* az) {
#pragma omp parallel for collapse(2)
	for (int l=0; l<3; l++) {
		for (int ir=0; ir<orbs->grid.n[iR]; ir++) {
			dUeedr(ir, l) = orbs->grid.d_dr(&Uee(0, l), ir);
		}
	}

    auto func_0 = [&](int ir, int il, int m) -> double {
		double r = orbs->grid.r(ir);
        return Uee(ir, 1) / r;
    };

    auto func_1 = [&](int ir, int il, int m) -> double {
        return atom_cache.dudz(ir) + dUeedr(ir, 0);
    };

    auto func_2 = [&](int ir, int il, int m) -> double {
		double r = orbs->grid.r(ir);
        return dUeedr(ir, 1) - Uee(ir, 1)/r;
    };

#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
        double az_local = - (field_E(field, t) + orbs->mpi_wf->norm(func_0) + orbs->mpi_wf->cos(func_1) + orbs->mpi_wf->cos2(func_2))*orbs->atom.orbs[orbs->mpi_rank].countElectrons;
		MPI_Gather(&az_local, 1, MPI_DOUBLE, az, 1, MPI_DOUBLE, 0, orbs->mpi_comm);
	} else
#endif
	{
		for (int ie=0; ie<orbs->atom.countOrbs; ++ie) {
            az[ie] = - (field_E(field, t) + orbs->wf[ie]->norm(func_0) + orbs->wf[ie]->cos(func_1) + orbs->wf[ie]->cos2(func_2))*orbs->atom.orbs[ie].countElectrons;
		}
	}
}

double calc_wf_ionization_prob(ShWavefunc const* wf) {
    auto func = [&](int ir, int il, int m) -> double {
        return mask_core(&wf->grid, ir, il, m);
    };
    return 1.0 - wf->norm(func);
}

template<class Grid>
double calc_orbs_ionization_prob(Orbitals<Grid> const* orbs) {
    auto func = [&](int ir, int il, int m) -> double {
        return mask_core(&orbs->grid, ir, il, m);
    };
    return orbs->atom.countElectrons - orbs->norm(func);
}

double calc_wf_jrcd(
		workspace::WfBase* ws,
		ShWavefunc* wf,
        AtomCache<ShGrid> const& atom_cache,
		field_t const* field,
		int Nt, 
		double dt,
		double t_smooth
) {
	double res = 0.0;
	double t = 0.0;
	double const t_max = Nt*dt;

	for (int i = 0; i < Nt; ++i) {
        res += calc_wf_az(wf, atom_cache, field, t)*smoothstep(t_max - t, 0, t_smooth);
        ws->prop(*wf, field, t, dt);
		t += dt;
	}

	return res*dt;
}

template<class Grid>
double calc_orbs_jrcd(
		workspace::OrbitalsWS<Grid>& ws,
		Orbitals<Grid>& orbs,
        AtomCache<Grid> const& atom_cache,
		field_t const* field,
		int Nt, 
		double dt,
		double t_smooth
) {
	double res = 0.0;
	double t = 0.0;
	double const t_max = Nt*dt;

	for (int i = 0; i < Nt; ++i) {
        res += calc_orbs_az(orbs, atom_cache, field, t)*smoothstep(t_max - t, 0, t_smooth);
        ws.prop(orbs, field, t, dt, true);
		t += dt;
	}

	return res*dt;
}

template double calc_wf_az_with_polarization<ShGrid>(Wavefunc<ShGrid> const* wf, AtomCache<ShGrid> const& atom_cache, double const Upol[], double const dUpol_dr[], field_t const* field, double t);
template double calc_wf_az_with_polarization<ShNotEqudistantGrid>(Wavefunc<ShNotEqudistantGrid> const* wf, AtomCache<ShNotEqudistantGrid> const& atom_cache, double const Upol[], double const dUpol_dr[], field_t const* field, double t);

template double calc_orbs_ionization_prob<ShGrid>(Orbitals<ShGrid> const* orbs);
template double calc_orbs_ionization_prob<ShNotEqudistantGrid>(Orbitals<ShNotEqudistantGrid> const* orbs);

template double calc_orbs_az<ShGrid>(Orbitals<ShGrid> const& orbs, AtomCache<ShGrid> const& atom_cache, field_t const* field, double t);
template double calc_orbs_az<ShNotEqudistantGrid>(Orbitals<ShNotEqudistantGrid> const& orbs, AtomCache<ShNotEqudistantGrid> const& atom_cache, field_t const* field, double t);
  
template void calc_orbs_az_ne<ShGrid>(Orbitals<ShGrid> const* orbs, AtomCache<ShGrid> const& atom_cache, field_t const* field, double t, double* az);
template void calc_orbs_az_ne<ShNotEqudistantGrid>(Orbitals<ShNotEqudistantGrid> const* orbs, AtomCache<ShNotEqudistantGrid> const& atom_cache, field_t const* field, double t, double* az);

template double calc_orbs_jrcd<ShGrid>(workspace::OrbitalsWS<ShGrid>& ws, Orbitals<ShGrid>& orbs, AtomCache<ShGrid> const& atom, field_t const* field, int Nt, double dt, double t_smooth);
template double calc_orbs_jrcd<ShNotEqudistantGrid>(workspace::OrbitalsWS<ShNotEqudistantGrid>& ws, Orbitals<ShNotEqudistantGrid>& orbs, AtomCache<ShNotEqudistantGrid> const& atom, field_t const* field, int Nt, double dt, double t_smooth);


template void calc_orbs_az_ne_Vee_0<ShGrid>(Orbitals<ShGrid> const* orbs, Array2D<double>& Uee, Array2D<double>& dUeedr, const AtomCache<ShGrid>& atom_cache, field_t const* field, double t, double* az);
template void calc_orbs_az_ne_Vee_0<ShNotEqudistantGrid>(Orbitals<ShNotEqudistantGrid> const* orbs, Array2D<double>& Uee, Array2D<double>& dUeedr, const AtomCache<ShNotEqudistantGrid>& atom_cache, field_t const* field, double t, double* az);

template void calc_orbs_az_ne_Vee_1<ShGrid>             (Orbitals<ShGrid>              const* orbs, Array2D<double>& Uee, Array2D<double>& dUeedr, const AtomCache<ShGrid>&              atom_cache, field_t const* field, double t, double* az);
template void calc_orbs_az_ne_Vee_1<ShNotEqudistantGrid>(Orbitals<ShNotEqudistantGrid> const* orbs, Array2D<double>& Uee, Array2D<double>& dUeedr, const AtomCache<ShNotEqudistantGrid>& atom_cache, field_t const* field, double t, double* az);
