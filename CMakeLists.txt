project(tdse)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
set(CMAKE_CXX_STANDARD 11)
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

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(GSL)

find_package(LAPACK)
if (LAPACK_FOUND)
	include_directories(${LAPACK_C_INCLUDE_PATH})
endif()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
	include_directories(/share/intel/mkl/include)
	link_directories(/share/intel/mkl/lib/intel64)
endif()

add_subdirectory(src)
