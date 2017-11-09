#include "calc.h"
#include "utils.h"
#include "abs_pot.h"


double calc_wf_az(sh_wavefunc_t const* wf, atom_t const* atom, field_t const* field, double t) {
	double func(sh_grid_t const* grid, int ir, int il, int m) { return atom->dudz(atom, grid, ir); }
    return - field_E(field, t) - sh_wavefunc_cos(wf, func);
}

double calc_orbs_az(orbitals_t const* orbs, atom_t const* atom, field_t const* field, double t) {
	double func(sh_grid_t const* grid, int ir, int il, int m) { return atom->dudz(atom, grid, ir); }
	return - field_E(field, t)*atom_get_count_electrons(atom) - orbitals_cos(orbs, func);
}

void calc_orbs_az_ne(orbitals_t const* orbs, field_t const* field, double t, double az[orbs->atom->n_orbs]) {
	double func(sh_grid_t const* grid, int ir, int il, int m) { return orbs->atom->dudz(orbs->atom, grid, ir); }
#ifdef _MPI
	if (orbs->mpi_comm != MPI_COMM_NULL) {
		double az_local = - (field_E(field, t) + sh_wavefunc_cos(orbs->mpi_wf, func))*orbs->atom->n_e[orbs->mpi_rank];
		MPI_Gather(&az_local, 1, MPI_DOUBLE, az, 1, MPI_DOUBLE, 0, orbs->mpi_comm);
	} else
#endif
	{
		for (int ie=0; ie<orbs->atom->n_orbs; ++ie) {
			az[ie] = - (field_E(field, t) + sh_wavefunc_cos(orbs->wf[ie], func))*orbs->atom->n_e[ie];
		}
	}
}

double calc_wf_ionization_prob(sh_wavefunc_t const* wf) {
	return 1.0 - sh_wavefunc_norm(wf, mask_core);
}

double calc_orbs_ionization_prob(orbitals_t const* orbs) {
	return atom_get_count_electrons(orbs->atom) - orbitals_norm(orbs, mask_core);
}

double calc_wf_jrcd(
		ws_wf_t* ws,
		sh_wavefunc_t* wf,
		atom_t const* atom,
		field_t const* field,
		int Nt, 
		double dt,
		double t_smooth
) {
	double res = 0.0;
	double t = 0.0;
	double const t_max = Nt*dt;

	for (int i = 0; i < Nt; ++i) {
		res += calc_wf_az(wf, atom, field, t)*smoothstep(t_max - t, 0, t_smooth);
		ws_wf_prop(ws, wf, atom, field, t, dt);
		t += dt;
	}

	return res*dt;
}

double calc_orbs_jrcd(
		ws_orbs_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		field_t const* field,
		int Nt, 
		double dt,
		double t_smooth
) {
	double res = 0.0;
	double t = 0.0;
	double const t_max = Nt*dt;

	for (int i = 0; i < Nt; ++i) {
		res += calc_orbs_az(orbs, atom, field, t)*smoothstep(t_max - t, 0, t_smooth);
		ws_orbs_prop(ws, orbs, atom, field, t, dt, true);
		t += dt;
	}

	return res*dt;
}
