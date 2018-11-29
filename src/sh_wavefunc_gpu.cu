#include "sh_wavefunc_gpu.h"
#include "utils.h"
#include "types.h"

#include <cuda_runtime.h>


void ShWavefuncGPU::init() {
    if (data == nullptr) {
        cudaMalloc(&data, sizeof(cdouble)*grid->size());
        data_own = true;
    }

	cudaMalloc(&d_res, sizeof(double));
	cudaMalloc(&d_ur, sizeof(double)*grid->n[iR]);
	ur = new double[grid->n[iR]];
}


ShWavefuncGPU::ShWavefuncGPU(cdouble* data, ShGrid const* grid, int const m):
    grid(grid),
    data(data), data_own(false),
	m(m) {
		init();
}

ShWavefuncGPU::ShWavefuncGPU(ShWavefunc const& wf):
	grid(wf.grid),
	data(nullptr), data_own(true),
	m(wf.m) {
		init();
		cudaMemcpy(data, wf.data, grid->size()*sizeof(cdouble), cudaMemcpyHostToDevice);
}

ShWavefunc* ShWavefuncGPU::get() {
	auto wf = new ShWavefunc(grid, m);

	cudaMemcpy(wf->data, data, grid->size()*sizeof(cdouble), cudaMemcpyDeviceToHost);

	return wf;
}

ShWavefuncGPU::~ShWavefuncGPU() {
	if (data_own) {
		cudaFree(data);
	}

	cudaFree(d_res);
	cudaFree(d_ur);
	delete[] ur;
}

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
using namespace cub;

template<int BLOCK_THREADS>
__global__ void kernel_wf_cos(cuComplex const* wf, int m, double const* u, double* res, int Nr, int Nl, double dr) {
	typedef BlockReduce<double, BLOCK_THREADS> BlockReduceT;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	int in = threadIdx.x;

	double data = 0.0;

	for (int il = 0; il < Nl-1; ++il) {
		for (int ir = in; ir < Nr; ir+=BLOCK_THREADS) {
			data += u[ir]*clm(il, m)*real(wf[ir + il*Nr]*conj(wf[ir + (il+1)*Nr]));
		}
	}

	double aggregate = BlockReduceT(temp_storage).Sum(data);

	if (threadIdx.x == 0) {
		*res = aggregate;
	}
}

double ShWavefuncGPU::cos_func(sh_f func) const {
#pragma omp parallel for
	for (int ir=0; ir<grid->n[iR]; ir++) {
		ur[ir] = func(grid, ir, 0, m);
	}

	cudaMemcpy(d_ur, ur, sizeof(double)*grid->n[iR], cudaMemcpyHostToDevice);

	return this->cos(ur);
}

double ShWavefuncGPU::cos(double const* d_u) const {
	double res;

	const int BLOCK_THREADS = 1024;
	kernel_wf_cos<BLOCK_THREADS><<<1, BLOCK_THREADS>>>((cuComplex*)data, m, d_u, d_res, grid->n[iR], grid->n[iL], grid->d[iR]);

	cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);

	return res;
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

	cudaMalloc(&d_res, sizeof(double)*N);
	cudaMalloc(&d_ur, sizeof(double)*grid->n[iR]);
	ur = new double[grid->n[iR]];
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

	cudaMemcpy(d_ur, ur, sizeof(double)*grid->n[iR], cudaMemcpyHostToDevice);

	return this->cos(d_ur, res);
}

double* ShWavefuncArrayGPU::cos(double const* d_u, double* res) const {
	if (res == nullptr) {
		res = new double[N];
	}

	dim3 blockDim(1);
	dim3 gridDim(N);
	kernel_wf_array_cos<<<gridDim, blockDim>>>((cuComplex*)data, m, d_u, d_res, N, grid->n[iR], grid->n[iL], grid->d[iR]);

	cudaMemcpy(res, d_res, N*sizeof(double), cudaMemcpyDeviceToHost);

	return res;
}
