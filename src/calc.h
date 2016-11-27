#pragma once


#include "fields.h"

#include "sphere_wavefunc.h"
#include "sphere_kn.h"

#include "hydrogen.h"


void calc_a(int Nt, double a[Nt], sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t field);
