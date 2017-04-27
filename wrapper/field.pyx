import numpy as np

cdef class Field:
    def __dealloc__(self):
        field_free(self.cdata)

    def E(self, double t):
        return field_E(self.cdata, t)

cdef class TwoColorGaussField(Field):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
                        double freq=5.6e-2, double phase=0.0,
                        double tp=2*3.14/5.6e-2, double t0=0.0):
        self.cdata = two_color_gauss_field_alloc(E0, alpha, freq, phase, tp, t0)

cdef class TwoColorSinField(Field):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
                        double freq=5.6e-2, double phase=0.0,
                        double tp=2*3.14/5.6e-2, double t0=0.0):
        self.cdata = two_color_sin_field_alloc(E0, alpha, freq, phase, tp, t0)

cdef class TwoColorTrField(Field):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
                        double freq=5.6e-2, double phase=0.0,
                        double tp=2*3.14/5.6e-2, double t0=0.0):
        self.cdata = two_color_tr_field_alloc(E0, alpha, freq, phase, tp, t0)
