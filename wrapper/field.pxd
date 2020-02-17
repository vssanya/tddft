cdef extern from "fields.h":
    ctypedef double (*field_func_t)(void* self, double t)
    ctypedef double (*field_prop_t)(void* field)

    ctypedef struct field_t:
        field_func_t fE
        field_func_t fA
        field_prop_t pT

    double field_E(field_t* field, double t)
    double field_A(field_t* field, double t)
    double field_T(field_t* field)

    double field_E_from_A(void* field, double t)
    double field_func_zero(void* field, double t)


    ctypedef struct field_op_t:
        field_func_t fA
        field_func_t fE
        field_prop_t pT

        field_t* f1
        field_t* f2

    double field_mul_A(field_op_t* field, double t)
    double field_mul_E(field_op_t* field, double t)
    double field_sum_A(field_op_t* field, double t)
    double field_sum_E(field_op_t* field, double t)
    double field_op_T(field_op_t* field)

    ctypedef struct field_time_delay_t:
        field_func_t fA
        field_func_t fE
        field_prop_t pT

        field_t* f
        double delay

    double field_time_delay_A(field_op_t* field, double t)
    double field_time_delay_E(field_op_t* field, double t)
    double field_time_delay_T(field_op_t* field)

    ctypedef struct field_gauss_env_t:
        field_func_t fA
        field_func_t fE
        field_prop_t pT

        double tp
        double dI

    double field_gauss_env_A(field_gauss_env_t* field, double t)
    double field_gauss_env_E(field_gauss_env_t* field, double t)
    double field_gauss_env_T(field_gauss_env_t* field)


    ctypedef struct field_sin_env_t:
        field_func_t fA
        field_func_t fE
        field_prop_t pT

        double tp

    double field_sin_env_A(field_sin_env_t* field, double t)
    double field_sin_env_E(field_sin_env_t* field, double t)
    double field_sin_env_T(field_sin_env_t* field)

    ctypedef struct field_tr_env_t:
        field_func_t fA
        field_func_t fE
        field_prop_t pT

        double t_const
        double t_smooth

    double field_tr_env_A(field_tr_env_t* field, double t)
    double field_tr_env_E(field_tr_env_t* field, double t)
    double field_tr_env_T(field_tr_env_t* field)

    ctypedef struct field_tr_sin_env_t:
        field_func_t fA
        field_func_t fE
        field_prop_t pT

        double t_const
        double t_smooth

    double field_tr_sin_env_A(field_tr_sin_env_t* field, double t)
    double field_tr_sin_env_E(field_tr_sin_env_t* field, double t)
    double field_tr_sin_env_T(field_tr_sin_env_t* field)

    ctypedef struct field_const_env_t:
        field_func_t fA
        field_func_t fE
        field_prop_t pT

        double tp

    double field_const_env_A(field_const_env_t* field, double t)
    double field_const_env_E(field_const_env_t* field, double t)
    double field_const_env_T(field_const_env_t* field)

    ctypedef struct field_car_t:
            field_func_t fA
            field_func_t fE
            field_prop_t pT

            double E
            double freq
            double phase

    double field_car_A(field_car_t* field, double t)
    double field_car_E(field_car_t* field, double t)
    double field_car_T(field_car_t* field)

    ctypedef struct field_const_t:
            field_func_t fA
            field_func_t fE
            field_prop_t pT

            double A

    double field_const_A(field_const_t* field, double t)

    ctypedef struct field_base_t:
        field_func_t fE
        field_func_t fA
        field_prop_t pT

        double E0
        double alpha
        double freq
        double phase
        double tp
        double t0

    double two_color_gauss_field_E(field_base_t* data, double t)
    double two_color_gauss_dadt_field_E(field_base_t* data, double t)
    double two_color_sin_field_E(field_base_t* data, double t)
    double two_color_tr_field_E(field_base_t* data, double t)

cdef class Field:
    cdef field_t* cdata

cdef class FieldBase(Field):
    cdef field_t cfield

cdef class OpField(Field):
    cdef field_op_t cfield
    cdef public Field f1
    cdef public Field f2

cdef class MulField(OpField):
    pass

cdef class SumField(OpField):
    pass

cdef class TimeDelay():
    cdef public double delay

cdef class TimeDelayField(Field):
    cdef field_time_delay_t cfield
    cdef public Field f

cdef class GaussEnvField(Field):
    cdef field_gauss_env_t cfield

cdef class SinEnvField(Field):
    cdef field_sin_env_t cfield

cdef class SinEnvTpField(Field):
    cdef field_sin_env_t cfield

cdef class TrEnvField(Field):
    cdef field_tr_env_t cfield

cdef class TrSinEnvField(Field):
    cdef field_tr_sin_env_t cfield

cdef class ConstEnvField(Field):
    cdef field_const_env_t cfield

cdef class CarField(Field):
    cdef field_car_t cfield

cdef class ConstField(Field):
    cdef field_const_t cfield

cdef class TwoColorBaseField(Field):
    cdef field_base_t cfield
    cdef public double t_fwhm

cdef class TwoColorGaussField(TwoColorBaseField):
    pass

cdef class TwoColorGaussAField(TwoColorBaseField):
    pass

cdef class TwoColorSinField(TwoColorBaseField):
    pass

cdef class TwoColorTrField(TwoColorBaseField):
    pass

cdef class TwoColorTrSinField(TwoColorBaseField):
    pass
