#pragma once

typedef double (*field_func_t)(double t, void const* data);

typedef struct {
	field_func_t func;
	void* data;
} field_t;

inline double field_E(field_t field, double t) {
	return field.func(t, field.data);
}
void field_free(field_t field);

field_t two_color_pulse_field_alloc(
		double E0,
		double alpha,
		double freq,
		double phase,
		double tp,
		double t0
		);

field_t field_sin_alloc(
		double E0,
		double alpha,
		double freq,
		double phase,
		double tp,
		double t0
		);
