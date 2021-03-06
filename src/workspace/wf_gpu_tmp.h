#pragma once

#include "../fields.h"

#include "../grid.h"
#include "../sh_wavefunc_gpu.h"
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

#include "../pycuda-complex.hpp"
typedef pycuda::complex<double> cuComplex;

typedef void(*cuMatrix_f)(cuComplex[2]);
typedef double(*cuSh_f)(double r, int l, int m);

namespace workspace {
    class WfGPUBase {
		public:
            WfGPUBase(AtomCache const* atom_cache, ShGrid const* grid, UabsCache const* uabs_cache, int num_threads);

            virtual ~WfGPUBase ();

			/* 
			 * [1 + 0.5iΔtH(t+Δt/2)] Ψ(r, t+Δt) = [1 - 0.5iΔtH(t+Δt/2)] Ψ(r, t)
			 *
			 * exp(-0.5iΔtHang(l,m, t+Δt/2))
			 * @param E = E(t+dt/2)
			 * */
			void prop_ang(ShWavefuncGPU& wf, double dt, int l, double E);

            void prop_at(ShWavefuncGPU& wf, cdouble dt, double* Ur);
			void prop_mix(ShWavefuncGPU& wf, sh_f Al, double dt, int l);

			void prop_abs(ShWavefuncGPU& wf, double dt);
			/*!
			 * \f[U(r,t) = \sum_l U_l(r, t)\f]
			 * \param[in] Ul = \f[U_l(r, t=t+dt/2)\f]
			 * */
            void prop_common(ShWavefuncGPU& wf, cdouble dt, int l_max, double** Ul, cuSh_f* Ulfunc);

            void prop(ShWavefuncGPU& wf, field_t const* field, double t, double dt);
            void prop_without_field(ShWavefuncGPU& wf, double dt);
            void prop_img(ShWavefuncGPU& wf, double dt);

            ShGrid const* grid;
            UabsCache const* uabs_cache;

			cdouble* alpha;
            cdouble* betta;
            double* uabs_data;

			int num_threads;

            AtomCache const* atom_cache;
	};
}
