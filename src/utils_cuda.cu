#include <cuda_runtime.h>

__global__ void empty() {}

void benchmark_gpu_init() {
	cudaFree(0);
	empty<<<1,1>>>();
	cudaDeviceSynchronize();
}
