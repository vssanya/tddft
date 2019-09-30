#include "masks.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

template<typename Grid>
CoreMask<Grid>::~CoreMask() {
#ifdef WITH_CUDA
	if (gpu_data != nullptr) {
		cudaFree(gpu_data);
	}
#endif
}

template<typename Grid>
double* CoreMask<Grid>::getGPUData() {
#ifdef WITH_CUDA
	if (gpu_data == nullptr) {
		auto size = sizeof(double)*grid->n[iR];
		cudaMalloc(&gpu_data, size);

		double tmp[grid->n[iR]];
		for (int i=0; i<grid->n[iR]; i++) {
			tmp[i] = (*this)(i, 0, 0);
		}

		cudaMemcpy(gpu_data, tmp, size, cudaMemcpyHostToDevice);
	}

	return gpu_data;
#else
	assert(false);
#endif
}

template class CoreMask<ShGrid>;
template class CoreMask<ShNotEqudistantGrid>;
template class CoreMask<SpGrid2d>;
