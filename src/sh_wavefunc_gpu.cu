#include "sh_wavefunc_gpu.h"


ShWavefuncGPU::ShWavefuncGPU(cdouble *data, const ShGrid *grid, const int m):
    data(data),
    grid(grid),
    m(m),
    data_own(false) {
    if (data == nullptr) {
		cudaMalloc(&data, sizeof(cuComplex)*grid->size());
        data_own = true;
    }
}
