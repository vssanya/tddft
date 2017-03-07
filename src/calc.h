#pragma once

#include "fields.h"

#include "sh_wavefunc.h"
#include "orbitals.h"
#include "sh_workspace.h"
#include "atom.h"


double calc_wf_ionization_prob(sh_wavefunc_t const* wf);
double calc_orbs_ionization_prob(orbitals_t const* orbs);

// az(t) = - Ez(t) - <Ψ|dUdz|Ψ>
// @param dUdz - depends only r. It's dUdz/cos(\theta).
double calc_wf_az(
		sh_wavefunc_t const* wf,
		atom_t const* atom,
		field_t field,
		double t
);
double calc_orbs_az(
		orbitals_t const* orbs,
		atom_t const* atom,
		field_t field,
		double t
);

double calc_wf_jrcd(
		sh_workspace_t* ws,
		sh_wavefunc_t* wf,
		atom_t const* atom,
		field_t field,
		int Nt, 
		double dt,
		double t_smooth
);

double calc_orbs_jrcd(
		sh_orbs_workspace_t* ws,
		orbitals_t* orbs,
		atom_t const* atom,
		field_t field,
		int Nt, 
		double dt,
		double t_smooth
);
