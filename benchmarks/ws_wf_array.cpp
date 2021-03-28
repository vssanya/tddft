#include "abs_pot.h"
#include "atom.h"
#include "grid.h"
#include "sh_wavefunc_gpu.h"
#include "wavefunc/sh_2d.h"
#include <benchmark/benchmark.h>
#include <ctime>
#include <workspace/wf_array_gpu.h>

static void BM_WSArrayGpu_Abs(benchmark::State& state) {
	auto atom = HAtom();

	auto dt = 2e-2;
	auto dr = 2e-1;
	auto Rmax = 100.0;
	int n[2] = {int(Rmax/dr), 16};

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
		ws.prop_abs(&wf, dt);
	}
}
