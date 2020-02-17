#include "orbs_spin.h"

template <typename Grid>
workspace::OrbitalsSpinWS<Grid>::OrbitalsSpinWS(
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
		): workspace::OrbitalsWS<Grid>(sh_grid, sp_grid, atom_cache, uabs, ylm_cache, Uh_lmax, Uxc_lmax, Uxc, propAtType, gauge, num_threads) {
}

template class workspace::OrbitalsSpinWS<ShGrid>;
template class workspace::OrbitalsSpinWS<ShNotEqudistantGrid>;
