#pragma once

#ifdef _MPI
#include <mpi.h>
#include "types.h"

template <typename T>
inline decltype(MPI_DOUBLE) getMpiType();

template <>
inline decltype(MPI_DOUBLE) getMpiType<double>() {
	return MPI_DOUBLE;
}

template <>
inline decltype(MPI_DOUBLE) getMpiType<cdouble>() {
	return MPI_C_DOUBLE_COMPLEX;
}
#else
typedef int MPI_Comm;
#endif
