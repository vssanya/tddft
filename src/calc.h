#pragma once

#include "fields.h"

#include "sh_wavefunc.h"
#include "orbitals.h"
#include "workspace.h"
#include "atom.h"


double calc_wf_ionization_prob(ShWavefunc const* wf);
double calc_orbs_ionization_prob(Orbitals const* orbs);

// az(t) = - Ez(t) - <Ψ|dUdz|Ψ>
// @param dUdz - depends only r. It's dUdz/cos(\theta).
template<class Grid>
double calc_wf_az(
		Wavefunc<Grid> const* wf,
        AtomCache const& atom_cache,
		field_t const* field,
		double t
) {
    auto func = [&](Grid const* grid, int ir, int il, int m) -> double {
        return atom_cache.dudz(ir);
    };
    return - field_E(field, t) - wf->cos(func);
}

double calc_wf_az_with_polarization(
		ShWavefunc const* wf,
		AtomCache const& atom_cache,
		double const Upol[],
		double const dUpol_dr[],
		field_t const* field,
		double t
);

double calc_orbs_az(
		Orbitals  const& orbs,
        AtomCache const& atom_cache,
		field_t const* field,
		double t
);

void calc_orbs_az_ne(
		Orbitals const* orbs,
        AtomCache const& atom_cache,
		field_t const* field,
		double t,
		double* az
);

double calc_wf_jrcd(
		workspace::WfBase* ws,
		ShWavefunc* wf,
        AtomCache const& atom,
		field_t const* field,
		int Nt, 
		double dt,
		double t_smooth
);

double calc_orbs_jrcd(
		workspace::orbs& ws,
		Orbitals& orbs,
        AtomCache const& atom,
		field_t const* field,
		int Nt, 
		double dt,
		double t_smooth
);
