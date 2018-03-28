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
	class OrbsSpin: public workspace::orbs {
        OrbsSpin(
                AtomCache const* atom_cache,
                ShGrid const* sh_grid,
                SpGrid const* sp_grid,
                UabsCache const* uabs,
                YlmCache const* ylm_cache,
                int Uh_lmax, int Uxc_lmax,
                potential_xc_f Uxc,
                int num_threads);

		virtual void init();
        void calc_Uee(Orbitals const* orbs, int Uxc_lmax, int Uh_lmax);
	};
}
