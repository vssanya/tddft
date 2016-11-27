#include <stdlib.h>
#include <stdio.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "sphere_kn.h"
#include "jrcd.h"

#include "utils.h"

#include "fourier.h"

#include "hydrogen.h"
#include "abs_pot.h"
#include "fields.h"
#include "calc.h"

int main() {
	sphere_grid_t grid = {
		.Nr = 2000,
		.Nl = 80,
		.dr = 0.125
	};

	double const omega = 5.6e-2;
	double const E0 = 3.77e-2;
	double const T = 2.0*M_PI/omega;
	double const tp = T*2.05/2.67/(2*log(2));

	double dt = 0.025;
	int Nt = (int)(3*T/dt);

	sphere_kn_workspace_t* ws = sphere_kn_workspace_alloc(&grid, dt, hydrogen_U, Uabs);
	sphere_wavefunc_t* psi = hydrogen_ground(&grid);

	field_t field = two_color_pulse_field_alloc(E0, 0.0, omega, 0.0, tp, -1.5*T);

	double a[Nt];
	calc_a(Nt, a, ws, psi, field);

	for (int i = 0; i < Nt; ++i) {
		printf("%f\n", a[i]);
	}

	two_color_pulse_field_free(field);
	sphere_kn_workspace_free(ws);
	sphere_wavefunc_free(psi);
	
	return 0;
}
