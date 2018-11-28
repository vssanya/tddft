set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

set( LIB_GPU_SOURCES
	"main.cu"
	)

set( LIB_GPU_HEADERS
	"sh_wavefunc_gpu.h"
	"workspace/wf_gpu.h"
	"workspace/wf_array_gpu.h"
	)

set( LIB_SOURCES
	"workspace/wf.cpp"
	"workspace/wf_with_source.cpp"
	"workspace/orbs.cpp"
	"workspace/orbs_spin.cpp"
	"workspace/gps.cpp"
	"workspace/common_alg.cpp"
	"workspace/sfa.cpp"
	"workspace/wf_with_polarization.cpp"
	sh_wavefunc.cpp
	hartree_potential.cpp
	integrate.c
	utils.cpp
	abs_pot.cpp
	fields.cpp
	calc.cpp
	sphere_harmonics.cpp
	orbitals.cpp
	atom.cpp
	eigen.cpp
	linalg.cpp
	tdsfm.cpp
	"sfa/anal.c"
	"hartree_potential/hp_o3.cpp"
	"hartree_potential/hp_o5.cpp"
	"wavefunc/2d.cpp"
	"maxwell/1d.cpp"
	)

set( LIB_HEADERS
	workspace.h
	"workspace/wf.h"
	"workspace/wf_with_source.h"
	"workspace/orbs.h"
	"workspace/orbs_spin.h"
	"workspace/gps.h"
	"workspace/common_alg.h"
	"workspace/sfa.h"
	"workspace/wf_with_polarization.h"
	"maxwell/1d.h"

	fourier.h
	kn.h
	prog.h
	grid.h
	sh_wavefunc.h
	hartree_potential.h
	types.h
	units.h
	utils.h
	abs_pot.h
	fields.h
	calc.h
	sphere_harmonics.h
	orbitals.h
	atom.h
	eigen.h
	linalg.h
	tdsfm.h
	integrate.h
	"sfa/anal.h"

	"wavefunc/2d.h"
	"wavefunc/cartesian_2d.h"
	)

add_library( tdse SHARED ${LIB_SOURCES} ${LIB_HEADERS} )
target_link_libraries( tdse m lapacke ${GSL_LIBRARIES} ${CUDA_LIBRARIES})

if (CUDA_FOUND)
	cuda_add_library( tdse_gpu SHARED ${LIB_GPU_SOURCES} ${LIB_GPU_HEADERS})
	target_link_libraries( tdse_gpu m )
endif()
