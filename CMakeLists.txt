project(tdse)
include(ExternalProject)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/src)

find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI)
if (MPI_FOUND)
	include_directories(${MPI_C_INCLUDE_PATH})
	add_definitions(-D_MPI)
endif()

ExternalProject_Add(CUB
	GIT_REPOSITORY https://github.com/NVlabs/cub.git
	GIT_TAG v1.8.0
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
)

ExternalProject_Get_property(CUB SOURCE_DIR)
include_directories("${SOURCE_DIR}")

find_package(GSL)

find_package(LAPACK)
if (LAPACK_FOUND)
	include_directories(${LAPACK_C_INCLUDE_PATH})
	add_definitions(-DWITH_LAPACK)
endif()

find_package(CUDA)

if (CUDA_FOUND)
	add_definitions(-DWITH_CUDA)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
	include_directories(/share/intel/mkl/include)
	link_directories(/share/intel/mkl/lib/intel64)
endif()

include_directories(/share/vssanya/miniconda3/include)

add_subdirectory(src)
