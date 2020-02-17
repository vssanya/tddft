/*
 * =====================================================================================
 *
 *       Filename:  orbs_spin.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  20.02.2018 14:28:56
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "orbs.h"
#include <mpi.h>


namespace workspace {
	template <typename Grid>
	class OrbitalsSpinWS: public OrbitalsWS<Grid> {
        OrbitalsSpinWS(
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
				);
	};
}
