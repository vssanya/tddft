set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11 -Wall -Og")

set( LIB_SOURCES
  sh_workspace.c
  sh_wavefunc.c
  hartree_potential.c
	utils.c
	abs_pot.c
	grid.c
	fields.c
	calc.c
	sphere_harmonics.c
	orbitals.c
	atom.c
	eigen.c
	linalg.c
	"sfa/anal.c"
  "hartree_potential/hp_o3.c"
  "hartree_potential/hp_o5.c"
	)

set( LIB_HEADERS
  fourier.h
  kn.h
  prog.h
  grid.h
  sh_workspace.h
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
  "sfa/anal.h"
  "hartree_potential/hp_o5.h"
  )

add_library( tdse SHARED ${LIB_SOURCES} ${LIB_HEADERS} )
target_link_libraries( tdse m gsl gslcblas lapacke)
