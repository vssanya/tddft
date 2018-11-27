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

__global__ void kernel_wf_cos(cuComplex* wf, int m, double* u, double* res, int N, int Nr, int Nl, double dr) {
	int in = blockIdx.x*blockDim.x + threadIdx.x;

	double sum = 0.0;

	for (int il = 0; il < Nl-1; ++il) {
		for (int ir = in; ir < Nr; ir+=N) {
			sum += u[ir]*clm(il, m)*real(wf[ir + il*Nr]*conj(wf[ir + (il+1)*Nr]));
		}
	}

	u[in] = sum;

	if (in == 0) {
		sum = 0.0;
		for (int i = 0; i < N; i++) {
			sum += u[i];
		}

		res[0] = 2*sum*dr;
	}
}

double ShWavefuncGPU::cos_func(sh_f func) const {
#pragma omp parallel for
	for (int ir=0; ir<grid->n[iR]; ir++) {
		ur[ir] = func(grid, ir, 0, m);
	}

	return this->cos(ur);
}

double ShWavefuncGPU::cos(double const* u) const {
	double res;

	cudaMemcpy(d_ur, u, sizeof(double)*grid->n[iR], cudaMemcpyHostToDevice);

	int N = 1024;

	kernel_wf_cos<<<N/32, 32>>>((cuComplex*)data, m, d_ur, d_res, N, grid->n[iR], grid->n[iL], grid->d[iR]);

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

	return this->cos(ur, res);
}

double* ShWavefuncArrayGPU::cos(double const* u, double* res) const {
	if (res == nullptr) {
		res = new double[N];
	}

	cudaMemcpy(d_ur, u, sizeof(double)*grid->n[iR], cudaMemcpyHostToDevice);

	dim3 blockDim(1);
	dim3 gridDim(N);
	kernel_wf_array_cos<<<gridDim, blockDim>>>((cuComplex*)data, m, d_ur, d_res, N, grid->n[iR], grid->n[iL], grid->d[iR]);

	cudaMemcpy(res, d_res, N*sizeof(double), cudaMemcpyDeviceToHost);

	return res;
}
