#pragma once

#include <complex.h>
#include <functional>

#include "grid.h"

typedef double _Complex cdouble;

typedef std::function<double(ShGrid const* grid, int ir, int il, int m)> sh_f;
