set(BENCHMARK_SOURCES
	"array.cpp"
	)

add_executable(benchmarks ${BENCHMARK_SOURCES})
target_link_libraries(benchmarks tdse benchmark::benchmark ${MPI_C_LIBRARIES})
