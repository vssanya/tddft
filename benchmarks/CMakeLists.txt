set(BENCHMARK_SOURCES
	"array.cpp"
	)

if (CUDA_FOUND)
	set(BENCHMARK_GPU_SOURCES
		"wf_array_gpu.cpp"
		)
endif()

add_executable(benchmarks ${BENCHMARK_SOURCES})
target_link_libraries(benchmarks tdse benchmark::benchmark ${MPI_C_LIBRARIES})

add_executable(benchmarks_gpu ${BENCHMARK_GPU_SOURCES})
target_link_libraries(benchmarks tdse tdse_gpu benchmark::benchmark ${MPI_C_LIBRARIES})
