cdef extern from "fields.h":
    ctypedef double (*field_func_t)(void* self, double t)

    ctypedef struct field_t:
        field_func_t func

    ctypedef struct field_base_t:
        field_func_t func
        double E0
        double alpha
        double freq
        double phase
        double tp
        double t0

    field_t* two_color_gauss_field_alloc(
            double E0,
            double alpha,
            double freq,
            double phase,
            double tp,
            double t0
            )
    field_t* two_color_sin_field_alloc(
            double E0,
            double alpha,
            double freq,
            double phase,
            double tp,
            double t0
            )
    field_t* two_color_tr_field_alloc(
        double E0,
        double alpha,
        double freq,
        double phase,
        double tp,
        double t0
    )
    void field_free(field_t* field)
    double field_E(field_t* field, double t)

cdef class Field:
    cdef field_t* cdata

cdef class TwoColorBaseField(Field):
    cdef public double t_fwhm

cdef class TwoColorGaussField(TwoColorBaseField):
    pass

cdef class TwoColorSinField(TwoColorBaseField):
    pass

cdef class TwoColorTrField(TwoColorBaseField):
    pass
