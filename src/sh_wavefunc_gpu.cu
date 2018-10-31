#include "sh_wavefunc_gpu.h"
#include "utils.h"
#include "types.h"

#include <cuda_runtime.h>


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

	cudaMalloc(&d_res, sizeof(double)*N);
	cudaMalloc(&d_ur, sizeof(double)*grid->n[iR]);
	ur = new double[grid->n[iR]];
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

	cudaFree(d_res);
	cudaFree(d_ur);
	delete[] ur;
}

__global__ void kernel_wf_array_cos(cuComplex* wf_array, int m, double const* u, double* res, int N, int Nr, int Nl, double dr) {
	int in = blockIdx.x*blockDim.x + threadIdx.x;

	if (in < N) {
		double sum = 0.0;

		cuComplex* wf = &wf_array[Nr*Nl*in];

		for (int il = 0; il < Nl-1; ++il) {
			for (int ir = 0; ir < Nr; ++ir) {
				sum += u[ir]*clm(il, m)*real(wf[ir + il*Nr]*conj(wf[ir + (il+1)*Nr]));
			}
		}

		res[in] = 2*sum*dr;
	}
}

double* ShWavefuncArrayGPU::cos_func(sh_f func, double* res) const {
#pragma omp parallel for
	for (int ir=0; ir<grid->n[iR]; ir++) {
		ur[ir] = func(grid, ir, 0, m);
	}

	return this->cos(ur, res);
}

double* ShWavefuncArrayGPU::cos(double const* u, double* res) const {
	if (res == nullptr) {
		res = new double[N];
	}

	cudaMemcpy(d_ur, u, sizeof(double)*N, cudaMemcpyHostToDevice);

	dim3 blockDim(1);
	dim3 gridDim(N);
	kernel_wf_array_cos<<<gridDim, blockDim>>>((cuComplex*)data, m, d_ur, d_res, N, grid->n[iR], grid->n[iL], grid->d[iR]);

	cudaMemcpy(res, d_res, N*sizeof(double), cudaMemcpyDeviceToHost);

	return res;
}
