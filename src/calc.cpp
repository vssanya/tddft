#include "calc.h"
#include "utils.h"
#include "abs_pot.h"


double calc_wf_az(ShWavefunc const* wf, const AtomCache &atom_cache, field_t const* field, double t) {
    auto func = [atom_cache](ShGrid const* grid, int ir, int il, int m) -> double {
        return atom_cache.dudz(ir);
    };
    return - field_E(field, t) - wf->cos(func);
}

double calc_orbs_az(Orbitals const* orbs, const AtomCache &atom_cache, field_t const* field, double t) {
    auto func = [atom_cache](ShGrid const* grid, int ir, int il, int m) -> double {
        return atom_cache.dudz(ir);
    };
    return - field_E(field, t)*atom_cache.atom.countElectrons - orbs->cos(func);
}

void calc_orbs_az_ne(Orbitals const* orbs, const AtomCache& atom_cache, field_t const* field, double t, double* az) {
    auto func = [atom_cache](ShGrid const* grid, int ir, int il, int m) -> double {
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

double calc_wf_ionization_prob(ShWavefunc const* wf) {
    return 1.0 - wf->norm(mask_core);
}

double calc_orbs_ionization_prob(Orbitals const* orbs) {
    return orbs->atom.countElectrons - orbs->norm(mask_core);
}

double calc_wf_jrcd(
		workspace::WfBase* ws,
		ShWavefunc* wf,
        AtomCache const& atom_cache,
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

double calc_orbs_jrcd(
		workspace::orbs* ws,
		Orbitals* orbs,
        AtomCache const& atom_cache,
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
        ws->prop(orbs, field, t, dt, true);
		t += dt;
	}

	return res*dt;
}
