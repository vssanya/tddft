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
double calc_wf_az(
		ShWavefunc const* wf,
        AtomCache const& atom_cache,
		field_t const* field,
		double t
);

double calc_orbs_az(
		Orbitals const* orbs,
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
		workspace::orbs* ws,
		Orbitals* orbs,
        AtomCache const& atom,
		field_t const* field,
		int Nt, 
		double dt,
		double t_smooth
);
