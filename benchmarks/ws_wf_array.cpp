#include <ctime>
#include <benchmark/benchmark.h>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#include <abs_pot.h>
#include <atom.h>
#include <grid.h>
#include <sh_wavefunc_gpu.h>
#include <wavefunc/sh_2d.h>
#include <workspace/wf_array_gpu.h>
#include <utils_cuda.h>

class ShWfArrayGpuTask {
public:
  ShWfArrayGpuTask()
      : atom(HAtom()), dt(2e-2), n{256, 16}, Rmax(50.0), grid(n, Rmax),
        atomCache(atom, grid), uabs(2.5, 25, 3), uabsCache(uabs, &grid), N(1000),
        ws(&atomCache, &grid, &uabsCache, N), wf_gs(grid, 0) {
					srand((unsigned) time(nullptr));
					wf_gs.set_random();
				}

  double dt;
  int n[2];

  double Rmax;

  ShGrid grid;

  HAtom atom;
  AtomCache<ShGrid> atomCache;

  UabsMultiHump uabs;
  UabsCache uabsCache;

  int N;
  workspace::WfArrayGpu ws;

  ShWavefunc wf_gs;
};

static void BM_WSArrayGpu_Abs(benchmark::State& state) {
	auto task = ShWfArrayGpuTask();
	auto wf = ShWavefuncArrayGPU(task.wf_gs, task.N);

	cudaDeviceSynchronize();
	for (auto _ : state) {
		auto start = std::chrono::steady_clock::now();
		task.ws.prop_abs(&wf, task.dt, state.range(0));
		cudaDeviceSynchronize();
		auto end = std::chrono::steady_clock::now();

		auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
}

static void BM_WSArrayGpu_PropAt(benchmark::State& state) {
	auto task = ShWfArrayGpuTask();
	auto wf = ShWavefuncArrayGPU(task.wf_gs, task.N);

	cudaDeviceSynchronize();
	for (auto _ : state) {
		auto start = std::chrono::steady_clock::now();
		task.ws.prop_at(&wf, task.dt, state.range(0));
		cudaDeviceSynchronize();
		auto end = std::chrono::steady_clock::now();


		auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
}

static void BM_WSArrayGpu_Prop(benchmark::State& state) {
	auto task = ShWfArrayGpuTask();
	auto wf = ShWavefuncArrayGPU(task.wf_gs, task.N);

	auto grid_E = Grid1d(task.N);
	auto E = Array1D<double>(grid_E);
	E.set_random();

	cudaDeviceSynchronize();
	for (auto _ : state) {
		auto start = std::chrono::steady_clock::now();
		task.ws.prop(&wf, E.data, task.dt, state.range(0));
		cudaDeviceSynchronize();
		auto end = std::chrono::steady_clock::now();


		auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
}

BENCHMARK(BM_WSArrayGpu_Abs)->Unit(benchmark::kMillisecond)->UseManualTime()->RangeMultiplier(4)->Range(4, 512);
BENCHMARK(BM_WSArrayGpu_PropAt)->Unit(benchmark::kMillisecond)->UseManualTime()->RangeMultiplier(4)->Range(16, 512*2);
BENCHMARK(BM_WSArrayGpu_Prop)->Unit(benchmark::kMillisecond)->UseManualTime()->RangeMultiplier(4)->Range(16, 512*2);
BENCHMARK_MAIN();
