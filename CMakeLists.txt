project(tdse C)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

include_directories(${PROJECT_SOURCE_DIR}/src)

find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

find_package(GSL)

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
	include_directories(/share/intel/mkl/include)
	link_directories(/share/intel/mkl/lib/intel64)
endif()

add_subdirectory(src)
