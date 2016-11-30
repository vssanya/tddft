set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11 -Wall -O2")


set( LIB_SOURCES
	sphere_kn.c
	sphere_wavefunc.c
	hydrogen.c
	utils.c
	abs_pot.c
	fields.c
	calc.c
	)

set( LIB_HEADERS
    fourier.h
    jrcd.h
    kn.h
    prog.h
    sphere_grid.h
    sphere_kn.h
    sphere_wavefunc.h
    types.h
    units.h
    utils.h
	abs_pot.h
	fields.h
	calc.h
   )

set( MAIN main.c )

add_library( tdse SHARED ${LIB_SOURCES} ${LIB_HEADERS} )
add_executable( main ${MAIN} )

target_link_libraries( tdse m )
target_link_libraries( main tdse m )
