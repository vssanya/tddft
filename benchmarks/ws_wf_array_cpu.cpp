#include <ctime>
#include <chrono>
#include <benchmark/benchmark.h>

#include <abs_pot.h>
#include <atom.h>
#include <grid.h>
#include <wavefunc/sh_arr.h>
#include <wavefunc/sh_2d.h>
#include <workspace/wf_array.h>

class ShWfArrayTask {
public:
  ShWfArrayTask()
      : atom(HAtom()), dt(2e-2), n{256, 16}, Rmax(50.0), grid(n, Rmax),
        atomCache(atom, grid), uabs(2.5, 25, 3), uabsCache(uabs, &grid), N(1000),
        ws(grid, &atomCache, uabsCache, workspace::PropAtType::Odr4,
           workspace::Gauge::LENGTH, 0), wf_gs(grid, 0), wf(N, grid, nullptr, MPI_COMM_NULL) {
					srand((unsigned) time(nullptr));
					wf_gs.set_random();
					wf.set_all(&wf_gs);
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
  workspace::WfArray<ShGrid> ws;

	ShWavefunc wf_gs;
	ShWavefuncArray wf;
};

static void BM_WSArrayCpu_Abs(benchmark::State& state) {
	auto task = ShWfArrayTask();

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		task.ws.prop_abs(&task.wf, task.dt);
		auto end = std::chrono::high_resolution_clock::now();

		auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
}

static void BM_WSArrayCpu_Prop(benchmark::State& state) {
	auto task = ShWfArrayTask();

	auto grid_E = Grid1d(task.N);
	auto E = Array1D<double>(grid_E);
	E.set_random();

	for (auto _ : state) {
		task.ws.prop(&task.wf, E.data, task.dt);
	}
}

BENCHMARK(BM_WSArrayCpu_Abs);
BENCHMARK(BM_WSArrayCpu_Prop);
BENCHMARK_MAIN();
