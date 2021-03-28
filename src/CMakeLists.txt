set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native")

if (CUDA_FOUND)
	set( LIB_GPU_SOURCES
		"main.cu"
		)

	set( LIB_GPU_HEADERS
		"sh_wavefunc_gpu.h"
		"workspace/wf_gpu.h"
		"workspace/wf_array_gpu.h"
		)
endif()

set( LIB_SOURCES
	"workspace/wf.cpp"
	"workspace/wf_with_source.cpp"
	"workspace/wf_array.cpp"
	"workspace/orbs.cpp"
	"workspace/gps.cpp"
	"workspace/common_alg.cpp"
	"workspace/sfa.cpp"
	"workspace/wf_with_polarization.cpp"

	"wavefunc/2d.cpp"
	"wavefunc/sh_2d.cpp"
	"wavefunc/sh_3d.cpp"
	"wavefunc/sh_arr.cpp"

	"maxwell/1d.cpp"
	"maxwell/cyl_1d.cpp"
	"maxwell/3d.cpp"

	"hartree_potential.cpp"
	"utils.cpp"
	"abs_pot.cpp"
	"fields.cpp"
	"masks.cpp"
	"calc.cpp"
	"sphere_harmonics.cpp"
	"orbitals.cpp"
	"atom.cpp"
	"eigen.cpp"
	"linalg.cpp"
	"tdsfm.cpp"
	"sfa/anal.cpp"
	"hartree_potential/hp_o3.cpp"
	"hartree_potential/hp_o5.cpp"
	)

set( LIB_HEADERS
	"workspace.h"
	"workspace/wf.h"
	"workspace/wf_with_source.h"
	"workspace/wf_array.h"
	"workspace/orbs.h"
	"workspace/orbs_spin.h"
	"workspace/gps.h"
	"workspace/common_alg.h"
	"workspace/sfa.h"
	"workspace/wf_with_polarization.h"

	"maxwell/1d.h"
	"maxwell/3d.h"

	"wavefunc/2d.h"
	"wavefunc/sh_2d.h"
	"wavefunc/sh_arr.h"
	"wavefunc/sh_3d.h"
	"wavefunc/base.h"
	"wavefunc/cartesian_2d.h"

	"array.h"
	"fourier.h"
	"kn.h"
	"prog.h"
	"grid.h"

	"hartree_potential.h"
	"types.h"
	"units.h"
	"utils.h"
	"abs_pot.h"
	"fields.h"
	"calc.h"
	"sphere_harmonics.h"
	"orbitals.h"
	"atom.h"
	"eigen.h"
	"linalg.h"
	"tdsfm.h"
	"integrate.h"
	"sfa/anal.h"
	)

add_library( tdse SHARED ${LIB_SOURCES} ${LIB_HEADERS} )
if (CUDA_FOUND)
	target_link_libraries( tdse m ${LAPACK_LIBRARIES} gsl ${CUDA_LIBRARIES})
else()
	target_link_libraries( tdse m ${LAPACK_LIBRARIES} ${GSL_LIBRARIES})
endif()


if (CUDA_FOUND)
	cuda_add_library( tdse_gpu SHARED ${LIB_GPU_SOURCES} ${LIB_GPU_HEADERS})
	target_link_libraries( tdse_gpu m )
endif()
