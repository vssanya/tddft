#include "sh_wavefunc.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <omp.h>

#include "utils.h"
#include "integrate.h"


/*
inline double ShWavefunc_integrate_o3(ShWavefunc const* wf, func_wf_t func, int l_max) {
	double res = 0.0;
#pragma omp parallel for reduction(+:res)
	for (int il = 0; il < l_max; ++il) {
		int ir = 0;
		{
			//res += 6*func(wf, ir, il);
			res += 4*func(wf, ir, il) + func(wf, ir+1, il);
		}
		for (ir = 2; ir < wf->grid->n[iR]-1; ir+=2) {
			res += func(wf, ir-1, il) + 4*func(wf, ir, il) + func(wf, ir+1, il);
		}

		if (ir != wf->grid->n[iR]-2) {
			ir = wf->grid->n[iR]-2;
			res += (func(wf, ir, il) + func(wf, ir+1, il))*3*0.5;
		}
	}
	return res*wf->grid->d[iR]/3;
}

inline double ShWavefunc_integrate_o4(ShWavefunc const* wf, func_wf_t func, int l_max) {
	double res = 0.0;
#pragma omp parallel for reduction(+:res)
	for (int il = 0; il < l_max; ++il) {
		int ir = 0;
		{
			int ir = 1;
			res += 32*func(wf, ir-1, il) + 12*func(wf, ir, il) + 32*func(wf, ir+1, il) + 7*func(wf, ir+2, il);
		}
		for (ir = 5; ir < wf->grid->n[iR]-3; ir+=4) {
			res += 7*func(wf, ir-2, il) + 32*func(wf, ir-1, il) + 12*func(wf, ir, il) + 32*func(wf, ir+1, il) + 7*func(wf, ir+2, il);
		}

		for (ir -= 2; ir < wf->grid->n[iR]-1; ir++) {
			res += (func(wf, ir, il) + func(wf, ir+1, il))*0.5*90.0/4.0;
		}
	}
	return 4.0*res*wf->grid->d[iR]/90.0;
}

inline double ShWavefunc_integrate_r2(ShWavefunc const* wf, func_wf_t func, int l_max, int Z) {
	double res = 0.0;

#pragma omp parallel for reduction(+:res)
	for (int il = 0; il < l_max; ++il) {
		int ir = 0;
		{
			if (il == 0) {
				double c = -Z*4*wf->grid->d[iR];
				res += ((16*c + 6)*func(wf, ir, il) + 2*c*func(wf, ir+1, il))/(1+3*c);
			} else {
				res += 6*func(wf, ir, il);
			}
		}
		for (ir = 2; ir < wf->grid->n[iR]-1; ir+=2) {
			res += func(wf, ir-1, il) + 4*func(wf, ir, il) + func(wf, ir+1, il);
		}

		if (ir != wf->grid->n[iR]-2) {
			ir = wf->grid->n[iR]-2;
			res += (func(wf, ir, il) + func(wf, ir+1, il))*3*0.5;
		}
	}
	return res*wf->grid->d[iR]/3;
}
*/
