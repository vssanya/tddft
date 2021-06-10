set(BENCHMARK_SOURCES
	"array.cpp"
	)

if (CUDA_FOUND)
	set(BENCHMARK_GPU_SOURCES
		"ws_wf_array.cpp"
		#"simple_gpu.cpp"
		)
endif()

add_executable(benchmarks ${BENCHMARK_SOURCES})
target_link_libraries(benchmarks tdse benchmark::benchmark ${MPI_C_LIBRARIES})

add_executable(bm_wf_array_cpu "ws_wf_array_cpu.cpp")
target_link_libraries(bm_wf_array_cpu tdse lapacke mpi_cxx benchmark::benchmark ${MPI_C_LIBRARIES})

if (CUDA_FOUND)
	cuda_add_executable(benchmarks_gpu ${BENCHMARK_GPU_SOURCES})
	target_link_libraries(benchmarks_gpu tdse tdse_gpu benchmark::benchmark ${MPI_C_LIBRARIES})
endif()
