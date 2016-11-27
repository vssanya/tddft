#pragma once

// Potential
double hydrogen_U(double r) __attribute__((pure));
double hydrogen_dUdz(double r) __attribute__((pure));

#include "sphere_grid.h"
#include "sphere_wavefunc.h"
sphere_wavefunc_t* hydrogen_ground(sphere_grid_t const* grid);
