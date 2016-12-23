cdef class TwoColorPulseField:
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
                        double freq=5.6e-2, double phase=0.0,
                        double tp=2*3.14/5.6e-2, double t0=0.0):
        self.data = two_color_pulse_field_alloc(E0, alpha, freq, phase, tp, t0)

    def E(self, double t):
        return field_E(self.data, t)
