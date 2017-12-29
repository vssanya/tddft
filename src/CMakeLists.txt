set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

set( LIB_SOURCES
	"workspace/wf.cpp"
	"workspace/orbs.cpp"
	#"workspace/sae.cpp"
	"workspace/gps.cpp"
	"workspace/common_alg.cpp"
	"workspace/sfa.cpp"

	sh_wavefunc.cpp
	hartree_potential.cpp
	integrate.c
	utils.c
	abs_pot.c
	grid.c
	fields.c
	calc.cpp
	sphere_harmonics.cpp
	orbitals.cpp
	atom.cpp
	eigen.cpp
	linalg.cpp
	tdsfm.cpp
	"sfa/anal.c"
	"hartree_potential/hp_o3.c"
	"hartree_potential/hp_o5.c"

	"wavefunc/2d.cpp"
	)

set( LIB_HEADERS
	workspace.h
	"workspace/wf.h"
	"workspace/orbs.h"
	"workspace/sae.h"
	"workspace/gps.h"
	"workspace/common_alg.h"
	"workspace/sfa.h"

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
target_link_libraries( tdse m gsl gslcblas lapacke)
