import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import print_figure

from libc.stdlib cimport malloc, free


cdef class Field:
    def __dealloc__(self):
        field_free(self.cdata)

    def E(self, double t):
        return field_E(self.cdata, t)

    @property
    def T(self):
        return 2*np.pi/self.freq

    def _figure_data(self, format):
        t = self.get_t(self.T/100)
        E = np.ndarray(t.shape)
        for i in range(t.size):
            E[i] = self.E(t[i])

        fig, ax = plt.subplots()
        fig.set_size_inches((6,3))
        ax.plot(t, E)
        ax.set_xlabel('t, (a.u.)')
        ax.set_ylabel('E, (a.u.)')

        data = print_figure(fig, format)
        plt.close(fig)
        return data

    def _repr_png_(self):
        return self._figure_data('png')

cdef object FIELDS = {}

cdef double field_base_func(void* self, double t):
    return FIELDS[<size_t>self]._E(t)

cdef class FieldBase(Field):
    def __cinit__(self):
        self.cdata = <field_t*>malloc(sizeof(field_t))
        self.cdata.func = field_base_func
        FIELDS[<size_t>self.cdata] = self

    def _E(self, t):
        return 0.0

cdef class TwoColorBaseField(Field):
    @property
    def E0(self):
        return (<field_base_t*>self.cdata).E0

    @property
    def alpha(self):
        return (<field_base_t*>self.cdata).alpha

    @property
    def E1(self):
        return self.E0*self.alpha

    @property
    def freq(self):
        return (<field_base_t*>self.cdata).freq

    @property
    def phase(self):
        return (<field_base_t*>self.cdata).phase

    @property
    def tp(self):
        return (<field_base_t*>self.cdata).tp

cdef class TwoColorGaussField(TwoColorBaseField):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
            double freq=5.6e-2, double phase=0.0,
            double t_fwhm=2*3.14/5.6e-2, double t0=0.0):
        self.t_fwhm = t_fwhm

        tp = self.t_fwhm / np.sqrt(2*np.log(2))
        self.cdata = two_color_gauss_field_alloc(E0, alpha, freq, phase, tp, t0)

    def duration(self, double dI=1e7):
        return 2*np.sqrt(0.5*self.tp**2*np.log(dI))

    def get_t(self, double dt, double dI=1e7, int nT=0):
        dur = self.duration(dI)
        t0 = -0.5*dur
        return np.arange(t0, t0+dur+self.T*nT, dt)

cdef class TwoColorSinField(TwoColorBaseField):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
            double freq=5.6e-2, double phase=0.0,
            double t_fwhm=2*3.14/5.6e-2, double t0=0.0):
        self.t_fwhm = t_fwhm
        cdef double tp = t_fwhm/(1.0 - 2.0*np.arcsin(0.5**(1.0/4.0))/np.pi)
        self.cdata = two_color_sin_field_alloc(E0, alpha, freq, phase, tp, t0)

    def __init__(self, double E0=5.34e-2 , double alpha=0.1,
            double freq=5.6e-2, double phase=0.0,
            double t_fwhm=2*3.14/5.6e-2, double t0=0.0):
        pass

    def get_t(self, double dt, int nT=0):
        return np.arange(0, self.tp+self.T*nT, dt)

cdef class TwoColorTrField(TwoColorBaseField):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
            double freq=5.6e-2, double phase=0.0,
            double tp=2*3.14/5.6e-2, double t0=0.0):
        self.cdata = two_color_tr_field_alloc(E0, alpha, freq, phase, tp, t0)

    def get_t(self, dt):
        return np.arange(0, self.tp, dt)
