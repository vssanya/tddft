#include "grid.h"
#include "types.h"

#include "pycuda-complex.hpp"

typedef pycuda::complex<double> cuComplex;


class ShWavefuncGPU {
	public:
		ShGrid const* grid;

		cuComplex* data; //!< data[i + l*grid->Nr] = \f$\Theta_{lm}(r_i)\f$
		bool data_own; //!< кто выделил данные

		int m;         //!< is magnetic quantum number

		ShWavefuncGPU(cdouble* data, ShGrid const* grid, int const m);
		ShWavefuncGPU(ShGrid const* grid, int const m): ShWavefuncGPU(nullptr, grid, m) {}
		~ShWavefuncGPU();
};
