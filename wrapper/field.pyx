import numpy as np

cdef class Field:
    def __dealloc__(self):
        field_free(self.data)

    def E(self, double t):
        return field_E(self.data, t)

cdef class TwoColorPulseField(Field):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
                        double freq=5.6e-2, double phase=0.0,
                        double tp=2*3.14/5.6e-2, double t0=0.0):
        self.data = two_color_pulse_field_alloc(E0, alpha, freq, phase, tp, t0)

cdef class SinField(Field):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
                        double freq=5.6e-2, double phase=0.0,
                        double tp=2*3.14/5.6e-2, double t0=0.0):
        self.data = field_sin_alloc(E0, alpha, freq, phase, tp, t0)
