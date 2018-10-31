#pragma once

#include "grid.h"
#include "types.h"
#include "sh_wavefunc.h"


class ShWavefuncGPU {
	public:
		ShGrid const* grid;

        cdouble* data; //!< data[i + l*grid->Nr] = \f$\Theta_{lm}(r_i)\f$
		bool data_own; //!< кто выделил данные

		int m;         //!< is magnetic quantum number

		ShWavefuncGPU(cdouble* data, ShGrid const* grid, int const m);
		ShWavefuncGPU(ShGrid const* grid, int const m): ShWavefuncGPU(nullptr, grid, m) {}
		~ShWavefuncGPU();
};

class ShWavefuncArrayGPU {
	public:
		ShGrid const* grid;
		int N; // count wavefunctions

        cdouble* data; //!< data[i + l*grid->Nr] = \f$\Theta_{lm}(r_i)\f$
		bool data_own; //!< кто выделил данные

		int m;         //!< is magnetic quantum number

		double* d_res; //!< is device results
		double* d_ur; //!< is device useful memory
		double* ur; //!< is host useful memory

		ShWavefuncArrayGPU(ShWavefunc const& wf, int N);
		ShWavefuncArrayGPU(cdouble* data, ShGrid const* grid, int const m, int N);
		ShWavefuncArrayGPU(ShGrid const* grid, int const m, int N): ShWavefuncArrayGPU(nullptr, grid, m, N) {}
		~ShWavefuncArrayGPU();

		ShWavefunc* get(int in);

		double* cos_func(sh_f func, double* res = nullptr) const;
		double* cos(double const* u, double* res = nullptr) const;
};
