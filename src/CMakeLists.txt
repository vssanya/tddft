set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11 -Wall -O2")


set( LIB_SOURCES
	sphere_kn.c
	sphere_wavefunc.c
	hydrogen.c
	hartree_potential.c
	utils.c
	abs_pot.c
	grid.c
	fields.c
	calc.c
	sphere_harmonics.c
	ks_orbitals.c
	jrcd.c
	)

set( LIB_HEADERS
    fourier.h
    jrcd.h
    kn.h
    prog.h
    grid.h
    sphere_kn.h
    sphere_wavefunc.h
	hartree_potential.h
    types.h
    units.h
    utils.h
	abs_pot.h
	fields.h
	calc.h
	sphere_harmonics.h
	ks_orbitals.h
   )

set( MAIN main.c )

add_library( tdse SHARED ${LIB_SOURCES} ${LIB_HEADERS} )
add_executable( main ${MAIN} )

target_link_libraries( tdse m gsl gslcblas )
target_link_libraries( main tdse m gsl gslcblas )
