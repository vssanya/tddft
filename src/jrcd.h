#pragma once

#include "sphere_wavefunc.h"
#include "sh_workspace.h"
#include "utils.h"

/* 
 * jrcd = Ng \int_{0}^{T} az dt 
 * @return jrcd / Ng
 * */
double jrcd(
		sh_workspace_t* ws,
		sphere_wavefunc_t* wf,
		field_t E,
		sh_f dUdz,
		int Nt, 
		double dt,
		double t_smooth
);
