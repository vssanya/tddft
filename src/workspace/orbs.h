#pragma once

#include "wf.h"
#include "../orbitals.h"

#include "../hartree_potential.h"
#include "../utils.h"
#include "../types.h"

namespace workspace
{
	double const UXC_NORM_L[] = {sqrt(1.0/(4.0*M_PI)), sqrt(3.0/(4*M_PI)), sqrt(5.0/(4*M_PI))};

	template<typename Grid>
	class OrbitalsWS {
		public:
			enum class TimeApproxUeeType {
				SIMPLE = 1, // Uee(t+dt/2) = Uee(t)
				TWO_POINT   // Uee(t+dt/2) = (Uee(t) + Uee(t+dt)) / 2
			};

            OrbitalsWS(
					Grid      const& sh_grid,
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
			virtual ~OrbitalsWS();

			virtual void init();

            void prop_simple   (Orbitals<Grid>& OrbitalsWS, field_t const* field, double t, double dt, bool calc_uee);
            void prop_two_point(Orbitals<Grid>& OrbitalsWS, field_t const* field, double t, double dt, bool calc_uee);
            void prop          (Orbitals<Grid>& OrbitalsWS, field_t const* field, double t, double dt, bool calc_uee);

            void prop_img(Orbitals<Grid>& OrbitalsWS, double dt);
			void prop_ha(Orbitals<Grid>& OrbitalsWS, double dt);
			void calc_Uee(Orbitals<Grid> const& OrbitalsWS, int Uxc_lmax, int Uh_lmax, double* Uee = nullptr);

			void setTimeApproxUeeTwoPointFor(Orbitals<Grid> const& OrbitalsWS);

			workspace::WavefuncWS<Grid> wf_ws;

			double* Utmp;
			double* Utmp_local;

			int Uh_lmax;

			potential_xc_f Uxc;
			int Uxc_lmax;

			double* Uee;
			int lmax;

            Grid const& sh_grid;
            SpGrid const& sp_grid;
			double* uh_tmp;
			double* n_sp; // for root
			double* n_sp_local;

			YlmCache const& ylm_cache;

			TimeApproxUeeType timeApproxUeeType;

			Orbitals<Grid>* tmpOrb;
			double* tmpUee;
	};
} /* workspace */ 
