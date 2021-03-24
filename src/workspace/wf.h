#pragma once

#include <omp.h>

#include "../fields.h"

#include "../grid.h"
#include "../wavefunc/sh_2d.h"
#include "../orbitals.h"
#include "../abs_pot.h"
#include "../atom.h"

#include "../utils.h"
#include "../types.h"

#include "../linalg.h"
#include "common_alg.h"

#include "../array.h"
#include "../wavefunc/sh_3d.h"

/*! \file
 * Split-step method:
 * \f[ e^{(A + B)dt} = e^\frac{Adt}{2} e^{Bdt} e^\frac{Adt}{2} + \frac{1}{24}\left[A + 2B, [A,B]\right] dt^3 + O(dt^4) \f]
 *
 * \f[ C_1 = \left[\frac{d^2}{dr^2}, r\right] = 2\frac{d}{dr} \f]
 * \f[ C_2 = \left[\frac{d^2}{dr^2} + 2r, C_1\right] = [2r, C_1] = 2(\frac{d}{dr} - 1) \f]
 *
 * For \f$A = 1/2 d^2/dr^2\f$ and \f$B = r\cos{\theta}E\f$:
 * 
 * */

namespace workspace {
	enum class PropAtType { 
		Odr3 = 3, // simple Crank-Nicolson scheme
		Odr4  // Numerov scheme
	};

	enum class Gauge {
		LENGTH = 0,
		VELOCITY
	};

	template<class Grid>
    class WavefuncWS {
		public:
			typedef std::function<double(int ir, int il, int m)> sh_f;

			WavefuncWS( 
					Grid    const& grid,
					AtomCache<Grid> const* atom_cache,
					UabsCache const& uabs,
					PropAtType propAtType,
					Gauge gauge,
					int num_threads
				  );

			void set_atom_cache(AtomCache<Grid> const* atom_cache) {
				this->atom_cache = atom_cache;
			}

			virtual ~WavefuncWS() {
				delete[] alpha;
				delete[] betta;
			}

			/* 
			 * [1 + 0.5iΔtH(t+Δt/2)] Ψ(r, t+Δt) = [1 - 0.5iΔtH(t+Δt/2)] Ψ(r, t)
			 *
			 * exp(-0.5iΔtHang(l,m, t+Δt/2))
			 * @param E = E(t+dt/2)
			 * */
			void prop_ang(Wavefunc<Grid>& wf, double dt, int l, double E);

			/*
			 * \brief Расчет функции \f[\psi(t+dt) = exp(-iH_{at}dt)\psi(t)\f],
			 * с точностью O(dr^4)
			 *
			 * \f[H_{at} = -0.5\frac{d^2}{dr^2} + U(r, l)\f]
			 * \f[exp(iAdt) = \frac{1 - iA}{1 + iA} + O(dt^3)\f]
			 *
			 * \param[in,out] wf
			 *
			 */
			void prop_at_Odr4(Wavefunc<Grid>& wf, cdouble dt, sh_f Ul);

			/*
			 * \brief Расчет функции \f[\psi(t+dt) = exp(-iH_{at}dt)\psi(t)\f],
			 * с точностью O(dr^3)
			 *
			 * \f[H_{at} = -0.5\frac{d^2}{dr^2} + U(r, l)\f]
			 * \f[exp(iAdt) = \frac{1 - iA}{1 + iA} + O(dt^3)\f]
			 *
			 * \param[in,out] wf
			 *
			 */
			void prop_at_Odr3(Wavefunc<Grid>& wf, cdouble dt, sh_f Ul);

			void prop_at(Wavefunc<Grid>& wf, cdouble dt, sh_f Ul) {
				switch (propAtType) {
					case PropAtType::Odr3:
						prop_at_Odr3(wf, dt, Ul);
						break;
					case PropAtType::Odr4:
						prop_at_Odr4(wf, dt, Ul);
						break;
				}
			}

			void prop_mix(Wavefunc<Grid>& wf, sh_f Al, double dt, int l);

			virtual void prop_abs(Wavefunc<Grid>& wf, double dt);

			/*!
			 * \f[U(r,t) = \sum_l U_l(r, t)\f]
			 * \param[in] Ul = \f[U_l(r, t=t+dt/2)\f]
			 * */
			void prop_common(Wavefunc<Grid>& wf, cdouble dt, int l_max, sh_f* Ul, sh_f* Al = nullptr);

			void prop(Wavefunc<Grid>& wf, double E, double A, double dt);
			void prop(Wavefunc<Grid>& wf, field_t const* field, double t, double dt) {
				prop(wf, field_E(field, t+dt/2), -field_A(field, t + dt/2), dt);
			}

			void prop_without_field(Wavefunc<Grid>& wf, double dt);

			void prop_img(Wavefunc<Grid>& wf, double dt);

			Grid      const& grid;
			AtomCache<Grid> const* atom_cache;
			UabsCache const& uabs;

			cdouble* alpha;
			cdouble* betta;

			PropAtType propAtType;
			Gauge gauge;

			int num_threads;
	};

	template<class Grid>
    class Wavefunc3DWS {
		public:
			typedef std::function<double(int ir, int il, int m)> sh_f;

			Wavefunc3DWS( 
					Grid    const& grid,
					AtomCache<Grid> const& atom_cache,
					UabsCache const& uabs,
					PropAtType propAtType,
					int num_threads
				  );

			virtual ~Wavefunc3DWS() {
				delete alpha;
				delete betta;
			}

			void prop_at(ShWavefunc3D<Grid>& wf, cdouble dt);
			void prop_abs(ShWavefunc3D<Grid>& wf, double dt);

			void prop(ShWavefunc3D<Grid>& wf, field_t const* Fx, field_t const* Fy, double t, double dt);
			void prop_img(ShWavefunc3D<Grid>& wf, double dt);

			Grid      const& grid;
			AtomCache<Grid> const& atom_cache;
			UabsCache const& uabs;

			Array2D<cdouble>* alpha;
			Array2D<cdouble>* betta;

			PropAtType propAtType;

			int num_threads;
	};

	/*
	 * \brief Расчет функции \f[\psi(t+dt) = exp(-iH_{at}dt)\psi(t)\f],
	 * с точностью O(dr^4)
	 *
	 * \f[H_{at} = -0.5\frac{d^2}{dr^2} + U(r, l)\f]
	 * \f[exp(iAdt) = \frac{1 - iA}{1 + iA} + O(dt^3)\f]
	 *
	 * \f[H_{at} = -0.5 (g(\xi)\frac{d^2}{d\xi^2} + h(\xi)\frac{d}{d\xi}) + U(r, l)\f]
	 * \f[H_{at} = -0.5 (g(\xi)M_2^{-1} d_2 + h(\xi) M_1^{-1} d_1) + U(r, l)\f]
	 *
	 * (M_1 M_2(1 - i U(r, l)) + 0.5 i (M_1))
	 *
	 * \param[in,out] wf
	 *
	 */
	template<>
	void WavefuncWS<ShNotEqudistantGrid>::prop_at_Odr4(Wavefunc<ShNotEqudistantGrid>& wf, cdouble dt, sh_f Ul);

	typedef WavefuncWS<ShGrid> WfBase;
	typedef WavefuncWS<ShNotEqudistantGrid> WfNeBase;
}
