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

namespace workspace {
	class OrbsSpin: public workspace::orbs {
        OrbsSpin(
                Atom const& atom,
                ShGrid const* sh_grid,
                SpGrid const* sp_grid,
                uabs_sh_t const* uabs,
                ylm_cache_t const* ylm_cache,
                int Uh_lmax, int Uxc_lmax,
                potential_xc_f Uxc,
                int num_threads): orbs(atom, sh_grid, sp_grid, uabs, ylm_cache, Uh_lmax, Uxc_lmax, Uxc, num_threads) {}

		virtual void init();
        void calc_Uee(Orbitals const* orbs, int Uxc_lmax, int Uh_lmax);
	};
}
