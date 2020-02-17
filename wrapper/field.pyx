import numpy as np
cimport numpy as np

import tdse.utils
if tdse.utils.is_jupyter_notebook():
    import matplotlib.pyplot as plt
    from IPython.core.pylabtools import print_figure

from libc.stdlib cimport malloc, free


ctypedef fused IDouble:
    double
    np.ndarray[double, ndim=1]

cdef class TimeDelay():
    def __init__(self, delay, units='au'):
        self.delay = tdse.utils.unit_to(delay, units, 'au')

ctypedef fused OpType:
    Field
    TimeDelay


cdef class Field:
    def E(self, IDouble t):
        cdef int i = 0

        if IDouble is double:
            return field_E(self.cdata, t)
        else:
            E = np.zeros(t.size)
            for i in range(t.size):
                E[i] = field_E(self.cdata, t[i])
            return E

    def A(self, IDouble t):
        cdef int i = 0

        if IDouble is double:
            return field_A(self.cdata, t)
        else:
            A = np.zeros(t.size)
            for i in range(t.size):
                A[i] = field_A(self.cdata, t[i])
            return A

    def __mul__(self, Field other):
        return MulField(self, other)

    def __add__(self, other):
        if isinstance(other, Field):
            return SumField(self, other)
        elif isinstance(other, TimeDelay):
            return TimeDelayField(self, other.delay)
        else:
            assert(False)

    @property
    def T(self):
        return field_T(self.cdata)

    @property
    def min_scale_t(self):
        return self.T

    def get_t(self, double dt, double dT=0):
        return np.arange(0, self.T+dT, dt)

    def _figure_data(self, format):
        t = self.get_t(self.min_scale_t/10)
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

    def _repr_latex_A_(self):
        return "?"

    def _repr_latex_E_(self):
        return "?"

    def _repr_latex(self):
        return r"$\vec{A}(t) = \vec{e}_z \left(" + self._repr_latex_A_() + r"\right)$"

cdef object FIELDS = {}

cdef double field_base_func(void* self, double t):
    return FIELDS[<size_t>self]._func(t)

cdef class FieldBase(Field):
    def __cinit__(self):
        self.cfield.fE = <field_func_t>field_base_func
        self.cfield.fA = <field_func_t>field_base_func
        self.cdata = &self.cfield

        FIELDS[<size_t>self.cdata] = self

    def _func(self, t):
        return 0.0

cdef class FieldBaseFromA(FieldBase):
    def __cinit__(self):
        self.cfield.fE = <field_func_t>field_E_from_A

    def _func(self, t):
        return 0.0


cdef class OpField(Field):
    def __cinit__(self, Field f1, Field f2):
        self.f1 = f1
        self.f2 = f2

        self.cfield.f1 = f1.cdata
        self.cfield.f2 = f2.cdata
        self.cfield.pT = <field_prop_t>field_op_T

        self.cdata = <field_t*>(&self.cfield)

cdef class TimeDelayField(Field):
    def __cinit__(self, Field f, double delay):
        self.f = f
        self.cfield.f = f.cdata
        self.cfield.delay = delay
        self.cfield.fE = <field_func_t>field_time_delay_E
        self.cfield.fA = <field_func_t>field_time_delay_A
        self.cfield.pT = <field_prop_t>field_time_delay_T

        self.cdata = <field_t*>(&self.cfield)

cdef class MulField(OpField):
    def __cinit__(self, Field f1, Field f2):
        self.cfield.fA = <field_func_t>field_mul_A
        self.cfield.fE = <field_func_t>field_mul_E

    def _repr_latex_A_(self):
        return r"{} \cdot {}".format(self.f1._repr_latex_A_(), self.f2._repr_latex_A_())

    def _repr_latex_E_(self):
        return r"{} \cdot {}".format(self.f1._repr_latex_E_(), self.f2._repr_latex_E_())


cdef class SumField(OpField):
    def __cinit__(self, Field f1, Field f2):
        super().__init__(f1, f2)
        self.cfield.fA = <field_func_t>field_sum_A
        self.cfield.fE = <field_func_t>field_sum_E

    def _repr_latex_A_(self):
        return r"{} + {}".format(self.f1._repr_latex_A_(), self.f2._repr_latex_A_())

    def _repr_latex_E_(self):
        return r"{} + {}".format(self.f1._repr_latex_E_(), self.f2._repr_latex_E_())

cdef class TwoColorBaseField(Field):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
            double freq=5.6e-2, double phase=0.0,
            double t_fwhm=2*3.14/5.6e-2, double t0=0.0, tp=None):
        self.cfield.E0 = E0
        self.cfield.alpha = alpha
        self.cfield.freq = freq
        self.cfield.phase = phase
        self.cfield.t0 = t0

        self.t_fwhm = t_fwhm

        self.cdata = <field_t*>(&self.cfield)

    @property
    def T(self):
        return 2.0*np.pi/self.freq

    @property
    def E0(self):
        return self.cfield.E0

    @property
    def alpha(self):
        return self.cfield.alpha

    @property
    def E1(self):
        return self.E0*self.alpha

    @property
    def freq(self):
        return self.cfield.freq

    @property
    def phase(self):
        return self.cfield.phase

    @property
    def tp(self):
        return self.cfield.tp

cdef class TwoColorGaussField(TwoColorBaseField):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
            double freq=5.6e-2, double phase=0.0,
            double t_fwhm=2*3.14/5.6e-2, double t0=0.0):
        self.cfield.tp = self.t_fwhm / np.sqrt(2*np.log(2))
        self.cfield.fA = <field_func_t>field_func_zero
        self.cfield.fE = <field_func_t>two_color_gauss_field_E

    def duration(self, double dI=1e7):
        return 2*np.sqrt(0.5*self.tp**2*np.log(dI))

    def get_t(self, double dt, double dI=1e7, int nT=0):
        dur = self.duration(dI)
        t0 = -0.5*dur
        return np.arange(t0, t0+dur+self.T*nT, dt)

cdef class TwoColorGaussAField(TwoColorBaseField):
    def __init__(self, double E0=5.34e-2 , double alpha=0.1,
            double freq=5.6e-2, double phase=0.0,
            double t_fwhm=2*3.14/5.6e-2, double t0=0.0):
        self.cfield.tp = self.t_fwhm / np.sqrt(2*np.log(2))
        self.cfield.fA = <field_func_t>field_func_zero
        self.cfield.fE = <field_func_t>two_color_gauss_dadt_field_E

    def duration(self, double Emin=1e-7):
        return 2*self.tp*np.sqrt(np.log(self.E0*(1.0+self.alpha)/Emin))

    def get_t(self, double dt, double Emin=1e-7, double dT=0.0):
        dur = self.duration(Emin)
        t0 = -0.5*dur
        return np.arange(t0, t0+dur+dT, dt)

cdef class TwoColorSinField(TwoColorBaseField):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1, double freq=5.6e-2,
            double phase=0.0, double t_fwhm=2*3.14/5.6e-2, double t0=0.0, tp = None):
        if tp is not None:
            self.cfield.tp = tp
        else:
            self.cfield.tp = t_fwhm/(1.0 - 2.0*np.arcsin(0.5**(1.0/4.0))/np.pi)

        self.cfield.fA = <field_func_t>field_func_zero
        self.cfield.fE = <field_func_t>two_color_sin_field_E

    def __init__(self, double E0=5.34e-2 , double alpha=0.1, double freq=5.6e-2,
            double phase=0.0, double t_fwhm=2*3.14/5.6e-2, double t0=0.0, double tp = 0.0):
        pass

    def get_t(self, double dt, int dT=0):
        return np.arange(0, self.tp+dT, dt)

cdef class TwoColorTrField(TwoColorBaseField):
    def __cinit__(self, double E0=5.34e-2 , double alpha=0.1,
            double freq=5.6e-2, double phase=0.0,
            double tp=2*3.14/5.6e-2, double t0=0.0):
        self.cfield.tp = tp
        self.cfield.fA = <field_func_t>field_func_zero
        self.cfield.fE = <field_func_t>two_color_tr_field_E

    def get_t(self, dt):
        return np.arange(0, self.tp, dt)

cdef class GaussEnvField(Field):
    def __init__(self, double t_fwhm, dI = None, T = None):
        self.cfield.tp = t_fwhm / np.sqrt(2*np.log(2))

        if T is not None:
            dI = np.exp(-2*(T/(2*self.cfield.tp))**2)

        self.cfield.dI = dI

        self.cfield.fA = <field_func_t>field_gauss_env_A
        self.cfield.fE = <field_func_t>field_gauss_env_E
        self.cfield.pT = <field_prop_t>field_gauss_env_T

        self.cdata = <field_t*>(&self.cfield)

    def _repr_latex_A_(self):
        return r"\exp(-\frac{t^2}{t_p^2})"

cdef class SinEnvField(Field):
    def __init__(self, double t_fwhm):
        self.cfield.tp = t_fwhm/(1.0 - 2.0*np.arcsin(0.5**(1.0/4.0))/np.pi)
        self.cfield.fA = <field_func_t>field_sin_env_A
        self.cfield.fE = <field_func_t>field_sin_env_E
        self.cfield.pT = <field_prop_t>field_sin_env_T

        self.cdata = <field_t*>(&self.cfield)

    def _repr_latex_A_(self):
        return r"\sin(\frac{\pi t}{t_p})^2"

cdef class SinEnvTpField(Field):
    def __init__(self, double tp):
        self.cfield.tp = tp
        self.cfield.fA = <field_func_t>field_sin_env_A
        self.cfield.fE = <field_func_t>field_func_zero
        self.cfield.pT = <field_prop_t>field_sin_env_T

        self.cdata = <field_t*>(&self.cfield)

    def _repr_latex_A_(self):
        return r"\sin(\frac{\pi t}{t_p})^2"

cdef class TrEnvField(Field):
    def __init__(self, double t_const, double t_smooth):
        self.cfield.t_const = t_const
        self.cfield.t_smooth = t_smooth
        self.cfield.fA = <field_func_t>field_tr_env_A
        self.cfield.fE = <field_func_t>field_E_from_A
        self.cfield.pT = <field_prop_t>field_tr_env_T

        self.cdata = <field_t*>(&self.cfield)

cdef class TrSinEnvField(Field):
    def __init__(self, double t_const, double t_smooth):
        self.cfield.t_const = t_const
        self.cfield.t_smooth = t_smooth
        self.cfield.fA = <field_func_t>field_tr_env_A
        self.cfield.fE = <field_func_t>field_tr_env_E
        self.cfield.pT = <field_prop_t>field_tr_env_T

        self.cdata = <field_t*>(&self.cfield)

cdef class ConstEnvField(Field):
    def __init__(self, double tp):
        self.cfield.tp = tp
        self.cfield.fA = <field_func_t>field_const_env_A
        self.cfield.fE = <field_func_t>field_const_env_E
        self.cfield.pT = <field_prop_t>field_const_env_T

        self.cdata = <field_t*>(&self.cfield)

cdef class CarField(Field):
    def __init__(self, double E, double freq, double phase):
        self.cfield.E = E
        self.cfield.freq = freq
        self.cfield.phase = phase
        self.cfield.fA = <field_func_t>field_car_A
        self.cfield.fE = <field_func_t>field_car_E
        self.cfield.pT = <field_prop_t>field_car_T

        self.cdata = <field_t*>(&self.cfield)

    def _repr_latex_A_(self):
        return r"E\sin(\omega t + \varphi)"

    def _repr_latex_E_(self):
        return r"-\frac{E}{\omega}\cos(\omega t + \varphi)"

    @property
    def freq(self):
        return self.cfield.freq

    @property
    def E0(self):
        return self.cfield.E

    @property
    def phase(self):
        return self.cfield.phase

cdef class ConstField(Field):
    def __init__(self, double A):
        self.cfield.A = A
        self.cfield.fA = <field_func_t>field_const_A
        self.cfield.fE = <field_func_t>field_func_zero
        self.cfield.pT = NULL

        self.cdata = <field_t*>(&self.cfield)

    def _repr_latex_A_(self):
        return r"A_0"

    def _repr_latex_E_(self):
        return None

def OneColorTrSinField(double E, double freq, double t_const, double t_smooth):
    return CarField(E, freq, 0.0)*TrSinEnvField(t_const, t_smooth)
