#pragma once

#include "fields.h"

#include "sh_wavefunc.h"
#include "orbitals.h"
#include "workspace.h"
#include "atom.h"
#include "array.h"


double calc_wf_ionization_prob(ShWavefunc const* wf);

template<class Grid>
double calc_orbs_ionization_prob(Orbitals<Grid> const* orbs);

// az(t) = - Ez(t) - <Ψ|dUdz|Ψ>
// @param dUdz - depends only r. It's dUdz/cos(\theta).
template<class Grid>
double calc_wf_az(
		Wavefunc<Grid> const* wf,
        AtomCache<Grid> const& atom_cache,
		field_t const* field,
		double t
) {
    auto func = [&](int ir, int il, int m) -> double {
        return atom_cache.dudz(ir);
    };
    return - field_E(field, t) - wf->cos(func);
}

template<class Grid>
double calc_wf_az(
		Wavefunc<Grid> const* wf_p,
		Wavefunc<Grid> const* wf_g,
        AtomCache<Grid> const& atom_cache,
		int l_max = -1
) {
    auto func = [&](int ir, int il, int m) -> double {
        return atom_cache.dudz(ir);
    };
    return creal(wf_p->cos(func, wf_g[0], l_max));
}

template<class Grid>
double calc_wf_az_with_polarization(
		Wavefunc<Grid> const* wf,
		AtomCache<Grid> const& atom_cache,
		double const Upol[],
		double const dUpol_dr[],
		field_t const* field,
		double t
);

template<class Grid>
double calc_orbs_az(
		Orbitals<Grid> const& orbs,
        AtomCache<Grid> const& atom_cache,
		field_t const* field,
		double t
);
  
template<class Grid>
void calc_orbs_az_ne(
		Orbitals<Grid> const* orbs,
        AtomCache<Grid> const& atom_cache,
		field_t const* field,
		double t,
		double* az
);

template<class Grid>
void calc_orbs_az_ne_Vee_0(Orbitals<Grid> const* orbs, Array2D<double>& Uee, Array2D<double>& dUeedr, const AtomCache<Grid>& atom_cache, field_t const* field, double t, double* az);

template<class Grid>
void calc_orbs_az_ne_Vee_1(Orbitals<Grid> const* orbs, Array2D<double>& Uee, Array2D<double>& dUeedr, const AtomCache<Grid>& atom_cache, field_t const* field, double t, double* az);

template<class Grid>
void calc_orbs_az_ne_Vee_2(Orbitals<Grid> const* orbs, Array2D<double>& Uee, Array2D<double>& dUeedr, const AtomCache<Grid>& atom_cache, field_t const* field, double t, double* az);

double calc_wf_jrcd(
		workspace::WfBase* ws,
		ShWavefunc* wf,
        AtomCache<ShGrid> const& atom,
		field_t const* field,
		int Nt, 
		double dt,
		double t_smooth
);

template<class Grid>
double calc_orbs_jrcd(
		workspace::OrbitalsWS<Grid>& ws,
		Orbitals<Grid>& orbs,
        AtomCache<Grid> const& atom,
		field_t const* field,
		int Nt, 
		double dt,
		double t_smooth
);

double calc_r_max(int N, double const E[], double dt, double r_atom);
double calc_pr_max(int N, double const E[], double dt, double r_max);
