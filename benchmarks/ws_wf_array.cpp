#include <ctime>
#include <chrono>
#include <benchmark/benchmark.h>

#include <abs_pot.h>
#include <atom.h>
#include <grid.h>
#include <sh_wavefunc_gpu.h>
#include <wavefunc/sh_2d.h>
#include <workspace/wf_array_gpu.h>

static void BM_WSArrayGpu_Abs(benchmark::State& state) {
	auto atom = HAtom();

	auto dt = 2e-2;
	auto Rmax = 50.0;
	int n[2] = {256, 16};

	auto grid = ShGrid(n, Rmax);
	auto atomCache = AtomCache(atom, grid);

	auto uabs = UabsMultiHump(2.5, 25, 3);
	auto uabsCache = UabsCache(uabs, &grid);

	int N = 1000;
	auto ws = workspace::WfArrayGpu(&atomCache, &grid, &uabsCache, N);

	auto wf_gs = ShWavefunc(grid, 0);
	srand((unsigned) time(nullptr));
	wf_gs.set_random();

	auto wf = ShWavefuncArrayGPU(wf_gs, N);

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		ws.prop_abs(&wf, dt);
		auto end = std::chrono::high_resolution_clock::now();

		auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
}

static void BM_WSArrayGpu_AbsTest(benchmark::State& state) {
	auto atom = HAtom();

	auto dt = 2e-2;
	auto Rmax = 50.0;
	int n[2] = {256, 16};

	auto grid = ShGrid(n, Rmax);
	auto atomCache = AtomCache(atom, grid);

	auto uabs = UabsMultiHump(2.5, 25, 3);
	auto uabsCache = UabsCache(uabs, &grid);

	int N = 1000;
	auto ws = workspace::WfArrayGpu(&atomCache, &grid, &uabsCache, N);

	auto wf_gs = ShWavefunc(grid, 0);
	srand((unsigned) time(nullptr));
	wf_gs.set_random();

	auto wf = ShWavefuncArrayGPU(wf_gs, N);

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		ws.prop_abs_test(&wf, dt);
		auto end = std::chrono::high_resolution_clock::now();

		auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
}

BENCHMARK(BM_WSArrayGpu_Abs)->UseManualTime();
BENCHMARK(BM_WSArrayGpu_AbsTest)->UseManualTime();
BENCHMARK_MAIN();
