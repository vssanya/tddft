#include "fields.h"

#include <stdlib.h>
#include <math.h>

#include "utils.h"


__attribute__((pure))
double two_color_fill(double E, double alpha, double phase, double tau) {
	return E*(cos(tau) + alpha*cos(2*tau + phase));
}

__attribute__((pure))
double gauss_env(double tp, double t) {
  return exp(-pow(t/tp, 2));
}

__attribute__((pure))
double sin_env(double tp, double t) {
  if (t < 0 || t > tp) {
    return 0.0;
  }
  return pow(sin(M_PI*t/tp), 2);
}

__attribute__((pure))
double tr_env(double tp, double t0, double t) {
  return smoothpulse(t, t0, tp-2*t0);
  /*
  if (t < 0 || t > tp) {
    return 0.0;
  } else if (t < t0) {
    return t/t0;
  } else if (t > tp - t0) {
    return (tp - t)/t0;
  } else {
    return 1.0;
  }
  */
}

__attribute__((pure))
double two_color_gauss_field_E(field_base_t const* data, double t) {
	double const tau = data->freq*(t - data->t0);
	return two_color_fill(data->E0, data->alpha, data->phase, tau)*gauss_env(data->tp, t - data->t0);
}

__attribute__((pure))
double two_color_sin_field_E(field_base_t const* data, double t) {
	double const tau = data->freq*(t-0.5*data->tp);
	return two_color_fill(data->E0, data->alpha, data->phase, tau)*sin_env(data->tp, t);
}

__attribute__((pure))
double two_color_tr_field_E(field_base_t const* data, double t) {
	double const tau = data->freq*(t - data->t0);
	return two_color_fill(data->E0, data->alpha, data->phase, tau)*tr_env(data->tp, data->t0, t);
}

field_t* field_base_alloc(field_func_t func, double E0, double alpha, double freq, double phase, double tp, double t0) {
	field_base_t* field = malloc(sizeof(field_base_t));

	field->func = func;
	field->E0 = E0;
	field->alpha = alpha;
	field->freq = freq;
	field->phase = phase;
	field->tp = tp;
	field->t0 = t0;

	return field;
}

field_t* two_color_gauss_field_alloc(double E0, double alpha, double freq, double phase, double tp, double t0) {
	return field_base_alloc((field_func_t)two_color_gauss_field_E, E0, alpha, freq, phase, tp, t0);
}

field_t* two_color_sin_field_alloc(double E0, double alpha, double freq, double phase, double tp, double t0) {
	return field_base_alloc((field_func_t)two_color_sin_field_E, E0, alpha, freq, phase, tp, t0);
}

field_t* two_color_tr_field_alloc(double E0, double alpha, double freq, double phase, double tp, double t0) {
	return field_base_alloc((field_func_t)two_color_tr_field_E, E0, alpha, freq, phase, tp, t0);
}

void field_free(field_t* field) {
	if (field != NULL) {
		free(field);
	}
}
