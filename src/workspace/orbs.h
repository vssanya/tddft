#pragma once

#include "wf.h"
#include "../orbitals.h"

#include "../hartree_potential.h"
#include "../utils.h"
#include "../types.h"
#include "../array.h"

#include <optional>
#include <vector>


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
					AtomCache<Grid> const* atom_cache,
					UabsCache const& uabs,
					YlmCache  const& ylm_cache,
					int Uh_lmax,
					int Uxc_lmax,
					XCPotentialEnum potentialType,
					PropAtType propAtType,
					Gauge gauge,
					int num_threads
					);
			virtual ~OrbitalsWS();

			virtual void init();

            void prop_simple(
					Orbitals<Grid>& OrbitalsWS,
					field_t const* field,
					double t, double dt,
					bool calc_uee,
					bool* activeOrbs = nullptr,
					int const dt_count[] = nullptr);

            void prop_two_point(
					Orbitals<Grid>& OrbitalsWS,
					field_t const* field,
					double t, double dt,
					bool calc_uee,
					bool* activeOrbs = nullptr,
					int const dt_count[] = nullptr);

            void prop(
					Orbitals<Grid>& OrbitalsWS,
					field_t const* field,
					double t, double dt,
					bool calc_uee,
					bool* activeOrbs = nullptr,
					int const dt_count[] = nullptr
			);

            void prop_img(
					Orbitals<Grid>& OrbitalsWS,
					double dt,
					bool activeOrbs[] = nullptr,
					int const dt_count[] = nullptr,
					bool calc_uee = true
			);

			void prop_ha(Orbitals<Grid>& OrbitalsWS, double dt,
					bool calc_uee = true, 
					bool activeOrbs[] = nullptr
					);

			void prop_abs(Orbitals<Grid>& OrbitalsWS, double dt, bool activeOrbs[] = nullptr);

			void calc_Uee(
					Orbitals<Grid> const& OrbitalsWS,
					int Uxc_lmax,
					int Uh_lmax,
					Array2D<double>* Uee = nullptr,
					std::optional<Range> rRange = std::nullopt
					);

			void setTimeApproxUeeTwoPointFor(Orbitals<Grid> const& OrbitalsWS);

			workspace::WavefuncWS<Grid> wf_ws;

			double* Utmp;
			double* Utmp_local;

			int Uh_lmax;

			CalcPotential<Grid>* calcPotential;
			int Uxc_lmax;

			Array2D<double>* Uee;
			int lmax;

            Grid const& sh_grid;
            SpGrid const& sp_grid;

			Grid2d uee_grid;

			double* uh_tmp;
			double* n_sp; // for root
			double* n_sp_local;

			YlmCache const& ylm_cache;

			TimeApproxUeeType timeApproxUeeType;

			Gauge gauge;

			Orbitals<Grid>* tmpOrb;
			Array2D<double>* tmpUee;
	};
} /* workspace */ 
