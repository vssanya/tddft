#pragma once

#include "sphere_wavefunc.h"
#include "sphere_kn.h"
#include "utils.h"

/* 
 * jrcd = Ng \int_{0}^{T} az dt 
 * @return jrcd / Ng
 * */
double jrcd(sphere_kn_workspace_t* ws, sphere_wavefunc_t* wf, field_t E, sphere_pot_t dUdz, int Nt);
