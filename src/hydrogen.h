#pragma once

#include "sphere_grid.h"

// Potential
double hydrogen_U(double r) __attribute__((pure));
double hydrogen_dUdz(double r) __attribute__((pure));
