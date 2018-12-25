#pragma once

#include "base.h"
#include "../utils.h"


template<class Grid>
class ShWavefunc3D: public WavefuncBase3D<Grid> {
	public:
		typedef double  (*func_wf_t        )(ShWavefunc3D const* wf, int ir, int il);
		typedef cdouble (*func_complex_wf_t)(ShWavefunc3D const* wf, int ir, int il);

		typedef std::function<double(int ir, int il, int m)> sh_f;


		ShWavefunc3D(cdouble* data, Grid const& grid): WavefuncBase3D<Grid>(data, grid) {}
		ShWavefunc3D(Grid const& grid): WavefuncBase3D<Grid>(grid) {}


		// <psi|U(r)cos(\theta)|psi>
		double cos(sh_f func) const {
			return 2*this->grid.template integrate<double>([this, func](int ir, int il, int im) -> double {
					return clm(il, im)*creal((*this)(ir, il, im)*conj((*this)(ir, il+1, im)))*func(ir, il, im);
					}, this->grid.n[iL]-1);
		}
};
