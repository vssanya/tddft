#include "sh_wavefunc_gpu.h"


ShWavefuncGPU::ShWavefuncGPU(cdouble *data, const ShGrid *grid, const int m):
    data(data),
    grid(grid),
    m(m),
    data_own(false) {
    if (data == nullptr) {
        cudaMalloc(&data, sizeof(cdouble)*grid->size());
        data_own = true;
    }
}

ShWavefuncGPU::~ShWavefuncGPU() {
	if (data_own) {
		cudaFree(data);
	}
}

ShWavefuncArrayGPU::ShWavefuncArrayGPU(cdouble* data, ShGrid const* grid, int const m, int N):
    grid(grid), N(N),
    data(data), data_own(false),
	m(m) {
    if (data == nullptr) {
        cudaMalloc(&data, sizeof(cdouble)*grid->size()*N);
        data_own = true;
    }
}

ShWavefuncArrayGPU::ShWavefuncArrayGPU(ShWavefunc const& wf, int N):
	grid(wf.grid), N(N),
	data(nullptr), data_own(true),
	m(wf.m) {
	cudaMalloc(&data, sizeof(cdouble)*grid->size()*N);
	for (int i = 0; i < N; i++) {
		cudaMemcpy(&data[grid->size()*i], wf.data, grid->size()*sizeof(cdouble), cudaMemcpyHostToDevice);
	}
}

ShWavefunc* ShWavefuncArrayGPU::get(int in) {
	auto wf = new ShWavefunc(grid, m);

	cudaMemcpy(wf->data, &data[in*grid->size()], grid->size()*sizeof(cdouble), cudaMemcpyDeviceToHost);

	return wf;
}

ShWavefuncArrayGPU::~ShWavefuncArrayGPU() {
	if (data_own) {
		cudaFree(data);
	}
}
