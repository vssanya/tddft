#pragma once

#include "wf.h"
#include "../orbitals.h"

#include "../hartree_potential.h"
#include "../utils.h"
#include "../types.h"

namespace workspace
{
	double const UXC_NORM_L[] = {sqrt(1.0/(4.0*M_PI)), sqrt(3.0/(4*M_PI)), sqrt(5.0/(4*M_PI))};

	class orbs {
		public:
			enum class TimeApproxUeeType {
				SIMPLE = 1, // Uee(t+dt/2) = Uee(t)
				TWO_POINT   // Uee(t+dt/2) = (Uee(t) + Uee(t+dt)) / 2
			};

            orbs(
					ShGrid    const& sh_grid,
					SpGrid    const& sp_grid,
					AtomCache const& atom_cache,
					UabsCache const& uabs,
					YlmCache  const& ylm_cache,
					int Uh_lmax,
					int Uxc_lmax,
					potential_xc_f Uxc,
					PropAtType propAtType,
					int num_threads
					);
			virtual ~orbs();

			virtual void init();

            void prop_simple   (Orbitals& orbs, field_t const* field, double t, double dt, bool calc_uee);
            void prop_two_point(Orbitals& orbs, field_t const* field, double t, double dt, bool calc_uee);
            void prop          (Orbitals& orbs, field_t const* field, double t, double dt, bool calc_uee);

            void prop_img(Orbitals& orbs, double dt);
			void prop_ha(Orbitals& orbs, double dt);
			void calc_Uee(Orbitals const& orbs, int Uxc_lmax, int Uh_lmax, double* Uee = nullptr);

			void setTimeApproxUeeTwoPointFor(Orbitals const& orbs);

			workspace::WfBase wf_ws;

			double* Utmp;
			double* Utmp_local;

			int Uh_lmax;

			potential_xc_f Uxc;
			int Uxc_lmax;

			double* Uee;
			int lmax;

            ShGrid const& sh_grid;
            SpGrid const& sp_grid;
			double* uh_tmp;
			double* n_sp; // for root
			double* n_sp_local;

			YlmCache const& ylm_cache;

			TimeApproxUeeType timeApproxUeeType;

			Orbitals* tmpOrb;
			double* tmpUee;
	};
} /* workspace */ 
