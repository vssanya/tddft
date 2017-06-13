#pragma once

typedef double (*field_func_t)(void const* self, double t);

typedef struct field_s {
	field_func_t func;
} field_t;

typedef struct {
  field_func_t func;

	double E0, alpha;
	double freq, phase;

	double tp, t0;
} field_base_t;

inline double field_E(field_t const* field, double t) {
	return field->func(field, t);
}
void field_free(field_t* field);

field_t* two_color_gauss_field_alloc(double E0, double alpha, double freq, double phase, double tp, double t0);
field_t* two_color_sin_field_alloc(double E0, double alpha, double freq, double phase, double tp, double t0);
field_t* two_color_tr_field_alloc(double E0, double alpha, double freq, double phase, double tp, double t0);
