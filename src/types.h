#pragma once

#include <complex.h>
#include <functional>

#include "grid.h"

typedef double _Complex cdouble;

typedef std::function<double(int ir, int il, int m)> sh_f;

#ifdef __CUDACC__
#include "pycuda-complex.hpp"
typedef pycuda::complex<double> cuComplex;
#endif
