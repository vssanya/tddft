#include "grid.h"
#include "types.h"

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
