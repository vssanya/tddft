#include <benchmark/benchmark.h>
#include <array.h>

static void BM_ArrayAdd(benchmark::State& state) {
	auto grid = Grid1d(10000);
	auto arr1 = Array1D<double>(grid);
	arr1.set(1.0);

	auto arr2 = Array1D<double>(grid);
	arr2.set(2.0);

	for (auto _ : state) {
		arr1.add(arr2);
	}
}

static void BM_ArrayAddSIMD(benchmark::State& state) {
	auto grid = Grid1d(10000);
	auto arr1 = Array1D<double>(grid);
	arr1.set(1.0);

	auto arr2 = Array1D<double>(grid);
	arr2.set(2.0);

	for (auto _ : state) {
		arr1.add_simd(arr2);
	}
}

BENCHMARK(BM_ArrayAdd);
BENCHMARK(BM_ArrayAddSIMD);
BENCHMARK_MAIN();
