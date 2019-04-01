#pragma once

#include "base.h"
#include "../utils.h"


template<typename Grid>
class ShWavefunc3D: public WavefuncBase3D<Grid> {
	public:
		typedef double  (*func_wf_t        )(ShWavefunc3D const* wf, int ir, int il);
		typedef cdouble (*func_complex_wf_t)(ShWavefunc3D const* wf, int ir, int il);

		typedef std::function<double(int ir, int il, int m)> sh_f;


		ShWavefunc3D(cdouble* data, Grid const& grid): WavefuncBase3D<Grid>(data, grid) {}
		ShWavefunc3D(Grid const& grid): WavefuncBase3D<Grid>(grid) {}

		Array1D<cdouble> slice(int l, int m) {
			return Array1D<cdouble>(&(*this)(0, l, m), Grid1d(this->grid.n[iR]));
		}

		// <psi|U(r)cos(\theta)|psi>
		double cos(sh_f func) const;

		// <psi| U(r) sin(\theta) sin(\varphi) |psi>
		cdouble sin_sin(sh_f func) const;

		// <psi| U(r) sin(\theta) sin(\varphi) |psi>
		cdouble sin_cos(sh_f func) const;
};

typedef ShWavefunc3D<ShNotEqudistantGrid3D> ShNeWavefunc3D;
