#include "fields.h"

#include <stdlib.h>
#include <math.h>
#include <algorithm>

#include "utils.h"

double field_E(field_t const* field, double t) {
	return field->fE(field, t);
}

double field_A(field_t const* field, double t) {
	return field->fA(field, t);
}

double field_T(field_t const* field) {
	if (field->pT != NULL) {
		return field->pT(field);
	}

	return 0.0;
}

double field_func_zero(void const* field, double t) {
	return 0.0;
}

double field_E_from_A(field_t const* field, double t) {
	const double dt = 1e-8;
	return -(field_A(field, t+dt) - field_A(field, t-dt))/(2*dt);
}


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
}

__attribute__((pure))
double tr_sin_env(double tp, double t_smooth, double t) {
	if (t < 0.0) {
		return 0.0;
	} else if (t < t_smooth) {
		return pow(sin(M_PI*t/(2*t_smooth)), 2);
	} else if (t < (tp - t_smooth)) {
		return 1.0;
	} else if (t < tp) {
		return 1.0 - pow(sin(M_PI*(t - tp + t_smooth)/(2*t_smooth)), 2);
	} else {
		return 0.0;
	}
}

__attribute__((pure))
double two_color_gauss_field_E(field_base_t const* data, double t) {
	double const tau = data->freq*(t - data->t0);
	return two_color_fill(data->E0, data->alpha, data->phase, tau)*gauss_env(data->tp, t - data->t0);
}

double one_color_gauss_dadt_E(double E, double freq, double t, double phase, double tp) {
	double tau = t*freq + phase;
	double tp_2 = pow(tp, 2);

	return E*(4.0*t*sin(tau)/(freq*tp_2) + (-1.0 + 4.0*pow(t, 2)/(pow(freq, 2)*pow(tp_2, 2)) - 2.0/(pow(freq, 2)*tp_2))*cos(tau))*exp(-pow(t/tp, 2));
}

double two_color_gauss_dadt_field_E(field_base_t const* data, double t) {
	return one_color_gauss_dadt_E(data->E0, data->freq, t, 0, data->tp) + one_color_gauss_dadt_E(data->alpha*data->E0, 2.0*data->freq, t, data->phase, data->tp);
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

double field_mul_A(field_op_t const* field, double t) {
	return field_A(field->f1, t)*field_A(field->f2, t);
}

double field_mul_E(field_op_t const* field, double t) {
	return field_E(field->f1, t)*field_A(field->f2, t) + field_A(field->f1, t)*field_E(field->f2, t);
}

double field_op_T(field_op_t const* field) {
    return std::max(field_T(field->f1), field_T(field->f2));
}

double field_sum_A(field_op_t const* field, double t) {
	return field_A(field->f1, t) + field_A(field->f2, t);
}

double field_sum_E(field_op_t const* field, double t) {
	return field_E(field->f1, t) + field_E(field->f2, t);
}

double field_time_delay_E(field_time_delay_t const* field, double t) {
	return field_E(field->f, t - field->delay);
}

double field_time_delay_A(field_time_delay_t const* field, double t) {
	return field_A(field->f, t - field->delay);
}

double field_time_delay_T(field_time_delay_t const* field) {
	return field_T(field->f) + field->delay;
}

double field_gauss_env_A(field_gauss_env_t const* field, double t) {
	t -= field_T((field_t*)field)*0.5;
	return exp(-pow(t/field->tp, 2));
}

double field_gauss_env_E(field_gauss_env_t const* field, double t) {
	double  dt = field_T((field_t*)field)*0.5;
	return 2.0*(t-dt)/pow(field->tp,2)*field_A((field_t*)field, t);
}

double field_gauss_env_T(field_gauss_env_t const* field) {
	return 2.0*field->tp*sqrt(-0.5*log(field->dI));
}

double field_sin_env_A(field_sin_env_t const* field, double t) {
  if (t < 0 || t > field->tp) {
    return 0.0;
  }
  return pow(sin(M_PI*t/field->tp), 2);
}

double field_sin_env_E(field_sin_env_t const* field, double t) {
  if (t < 0 || t > field->tp) {
    return 0.0;
  }
  return -2.0*sin(M_PI*t/field->tp)*cos(M_PI*t/field->tp)*M_PI/field->tp;
}

double field_sin_env_T(field_sin_env_t const* field) {
	return field->tp;
}

double field_tr_env_A(field_tr_env_t const* field, double t) {
  return smoothpulse(t, field->t_smooth, field->t_const);
}

double field_tr_env_E(field_tr_env_t const* field, double t) {
  return field_E_from_A((field_t*)field, t);
}

double field_tr_env_T(field_tr_env_t const* field) {
	return field->t_const + 2.0*field->t_smooth;
}

double field_tr_sin_env_A(field_tr_sin_env_t const* field, double t) {
	if (t < 0.0) {
		return 0.0;
	} else if (t < field->t_smooth) {
		return pow(sin(M_PI*t/(2*field->t_smooth)), 2);
	} else if (t < (field->t_const - field->t_smooth)) {
		return 1.0;
	} else if (t < field->t_const) {
		return 1.0 - pow(sin(M_PI*(t - field->t_const + field->t_smooth)/(2*field->t_smooth)), 2);
	} else {
		return 0.0;
	}
}

double field_tr_sin_env_E(field_tr_sin_env_t const* field, double t) {
	double freq = M_PI/(2*field->t_smooth);
	if (t < 0.0) {
		return 0.0;
	} else if (t < field->t_smooth) {
		return -2*freq*sin(freq*t)*cos(freq*t);
	} else if (t < (field->t_const - field->t_smooth)) {
		return 0.0;
	} else if (t < field->t_const) {
		t += - field->t_const + field->t_smooth;
		return 2*freq*sin(freq*t)*cos(freq*t);
	} else {
		return 0.0;
	}
}

double field_tr_sin_env_T(field_tr_sin_env_t const* field) {
	return field->t_const + 2.0*field->t_smooth;
}

double field_car_A(field_car_t const* field, double t) {
	return -field->E*sin(field->freq*t + field->phase)/field->freq;
}

double field_car_E(field_car_t const* field, double t) {
	return field->E*cos(field->freq*t + field->phase);
}

double field_car_T(field_car_t const* field) {
	return 2*M_PI/field->freq;
}

double field_const_env_A(field_const_env_t const* field, double t) {
	if (t < 0.0 || t > field->tp) {
		return 0.0;
	} else {
		return 1.0;
	}
}

double field_const_env_E(field_const_env_t const* field, double t) {
	return 0.0;
}

double field_const_env_T(field_const_env_t const* field) {
	return field->tp;
}

double field_const_A(field_const_t const* field, double t) {
	return field->A;
}
