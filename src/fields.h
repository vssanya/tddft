#pragma once

#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef double (*field_func_t)(void const* field, double t);
typedef double (*field_prop_t)(void const* field);
typedef struct field_t {
	field_func_t fA;
	field_func_t fE;
	field_prop_t pT;
} field_t;

double field_func_zero(void const* field, double t);
double field_E_from_A(field_t const* field, double t);

double field_E(field_t const* field, double t);
double field_A(field_t const* field, double t);
double field_T(field_t const* field);

#define FIELD_STRUCT(name, ...)\
	typedef struct field_##name##_t {\
		field_func_t fA;\
		field_func_t fE;\
		field_prop_t pT;\
		__VA_ARGS__\
	} field_##name##_t;

#define FIELD_METHODS(name, ...)\
	double field_##name##_A(field_##name##_t const* data, double t);\
	double field_##name##_E(field_##name##_t const* data, double t);\
	double field_##name##_T(field_##name##_t const* data);

#define FIELD(name, ...)\
	FIELD_STRUCT(name, __VA_ARGS__)\
	FIELD_METHODS(name, __VA_ARGS__)

typedef struct {
	field_func_t fA;
	field_func_t fE;
	field_prop_t pT;

	double E0, alpha;
	double freq, phase;

	double tp, t0;
} field_base_t;

double two_color_gauss_field_E(field_base_t const* data, double t);
double two_color_gauss_dadt_field_E(field_base_t const* data, double t);
double two_color_sin_field_E(field_base_t const* data, double t);
double two_color_tr_field_E(field_base_t const* data, double t);

FIELD_STRUCT( op,
		field_t const* f1;
		field_t const* f2;
		)

double field_mul_A(field_op_t const* field, double t);
double field_mul_E(field_op_t const* field, double t);
double field_sum_A(field_op_t const* field, double t);
double field_sum_E(field_op_t const* field, double t);
double field_op_T(field_op_t const* field);

FIELD( time_delay,
		field_t const* f;
		double delay;
		)

FIELD( gauss_env,
		double tp;
		double dI;
	 )

FIELD( sin_env,
		double tp;
	 )

FIELD( tr_env,
		double t_const;
		double t_smooth;
	 )

FIELD( tr_sin_env,
		double t_const;
		double t_smooth;
	 )

FIELD_STRUCT( const,
		double A;
	 )

double field_const_A(field_const_t const* field, double t);

FIELD( car,
		double E;
		double freq;
		double phase;
	 )

FIELD( car_chirp,
		double E;
		double freq;
		double phase;
		double chirp;
	 )

FIELD( const_env,
		double tp;
	 )

#ifdef __cplusplus
}
#endif
