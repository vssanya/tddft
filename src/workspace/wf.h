#pragma once

#include "../fields.h"

#include "../grid.h"
#include "../sh_wavefunc.h"
#include "../orbitals.h"
#include "../abs_pot.h"
#include "../atom.h"

#include "../utils.h"
#include "../types.h"

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
    class WfBase {
		public:
            WfBase(AtomCache const* atom_cache, ShGrid const* grid, uabs_sh_t const* uabs, int num_threads);

            virtual ~WfBase ();

			/* 
			 * [1 + 0.5iΔtH(t+Δt/2)] Ψ(r, t+Δt) = [1 - 0.5iΔtH(t+Δt/2)] Ψ(r, t)
			 *
			 * exp(-0.5iΔtHang(l,m, t+Δt/2))
			 * @param E = E(t+dt/2)
			 * */
			void prop_ang(ShWavefunc& wf, double dt, int l, double E);

            void prop_at(ShWavefunc& wf, cdouble dt, sh_f Ul);
			void prop_mix(ShWavefunc& wf, sh_f Al, double dt, int l);

			void prop_abs(ShWavefunc& wf, double dt);
			/*!
			 * \f[U(r,t) = \sum_l U_l(r, t)\f]
			 * \param[in] Ul = \f[U_l(r, t=t+dt/2)\f]
			 * */
            void prop_common(ShWavefunc& wf, cdouble dt, int l_max, sh_f* Ul, sh_f* Al = nullptr);

            void prop(ShWavefunc& wf, field_t const* field, double t, double dt);
            void prop_without_field(ShWavefunc& wf, double dt);
            void prop_img(ShWavefunc& wf, double dt);

            ShGrid const* grid;
			uabs_sh_t const* uabs;

			cdouble* alpha;
			cdouble* betta;

			int num_threads;

            AtomCache const* atom_cache;
	};

    class WfE: public WfBase {
		public:
            WfE(AtomCache const* atom_cache, ShGrid const* grid, uabs_sh_t const* uabs, int num_threads):
                WfBase(atom_cache, grid, uabs, num_threads) {}
            void prop(ShWavefunc& wf, field_t const* field, double t, double dt);
    };

    class WfA: public WfBase {
		public:
            WfA(AtomCache const* atom_cache, ShGrid const* grid, uabs_sh_t const* uabs, int num_threads):
                WfBase(atom_cache, grid, uabs, num_threads) {}
            void prop(ShWavefunc& wf, field_t const* field, double t, double dt);
    };
}

