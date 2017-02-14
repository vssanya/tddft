#include "fields.h"

#include <stdlib.h>
#include <math.h>


typedef struct {
	double E0, alpha;
	double freq, phase;

	double tp, t0;
} two_color_pulse_field_t;

double two_color_pulse_field_E(double t, two_color_pulse_field_t const* data) __attribute__((pure));
double two_color_pulse_field_E(double t, two_color_pulse_field_t const* data) {
	double const tau = data->freq*(t - data->t0);
	return data->E0*(cos(tau) + data->alpha*cos(2*tau + data->phase))* // fill
		   exp(-pow(t - data->t0, 2)/pow(data->tp, 2)); // env
}

field_t two_color_pulse_field_alloc(
		double E0,
		double alpha,
		double freq,
		double phase,
		double tp,
		double t0
		) {
	two_color_pulse_field_t* data = malloc(sizeof(two_color_pulse_field_t));
	data->E0 = E0;
	data->alpha = alpha;
	data->freq = freq;
	data->phase = phase;
	data->tp = tp;
	data->t0 = t0;

	return (field_t) {
		.func = (field_func_t)two_color_pulse_field_E,
		.data = data
	};
}

void two_color_pulse_field_free(field_t field) {
	free((two_color_pulse_field_t*) field.data);
}