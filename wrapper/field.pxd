cdef extern from "fields.h":
    ctypedef double (*field_func_t)(double t, void* data)
    ctypedef struct field_t:
        field_func_t func
        void* data
    field_t two_color_pulse_field_alloc(
            double E0,
            double alpha,
            double freq,
            double phase,
            double tp,
            double t0
            )
    void two_color_pulse_field_free(field_t field)
    double field_E(field_t field, double t)

cdef class Field:
    cdef field_t data

cdef class TwoColorPulseField(Field):
    pass
