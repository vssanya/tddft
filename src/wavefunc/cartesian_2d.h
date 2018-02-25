#pragma once

#include "../types.h"
#include "../grid.h"

#include "2d.h"


struct CtWavefunc : public Wavefunc2d {
    CtWavefunc(): Wavefunc2d() {}
    CtWavefunc(Grid2d const* grid): Wavefunc2d(grid) {}

	double norm() const;
};
