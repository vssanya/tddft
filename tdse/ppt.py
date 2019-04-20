import scipy.integrate as si
from scipy import LowLevelCallable

from cffi import FFI
from numba import cffi_support

import numba as nb
import numpy as np

src = """

/* Define the C struct */
typedef struct my_struct {
    double x;
    int m;
} my_struct;

/* Define a callback function */
typedef double (*my_func)(double, my_struct*);
"""


ffi = FFI()
ffi.cdef(src)

sig = cffi_support.map_type(ffi.typeof('my_func'), use_record_dtype=True)

my_struct = nb.types.Record.make_c_struct([
   ('m', nb.types.int64),
   ('x', nb.types.float64)
])

LOOKUP_TABLE = np.array(
        [
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000]
        , dtype='int64')

@nb.jit
def factorial(n):
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]

@nb.jit(nopython=True)
def calc_N(N0, w, dt):
    N = np.zeros(N0.size)
    
    N[0] = 0.0
    for i in range(1, N.size):
        N[i] = (N[i-1] + 0.5*(N0[i]*w[i] + N0[i-1]*w[i-1] - N[i-1]*w[i-1])*dt) / (1 + 0.5*w[i]*dt)
        
    return N

@nb.cfunc(sig)
def int_func(t, data):
    values = nb.carray(data, 1)
    return np.exp(-t*values[0].x**2)*t**values[0].m / np.sqrt(1 - t)

@nb.jit("float64(float64, int64)")
def w_m(x, m):
    mydata = ffi.new('my_struct[1]')
    ptr = ffi.cast('my_struct*', mydata)
    ptr[0].x = 1.0
    ptr[0].m = 0
    
    return 0.5*x**(2*m + 1)*si.quad(LowLevelCallable(int_func.ctypes, ptr), 0, 1)[0]

@nb.jit("float64(float64)")
def alpha(gamma):
    return 2*(np.arcsinh(gamma) - gamma/np.sqrt(1 + gamma**2))

@nb.jit("float64(float64)")
def betta(gamma):
    return 2*gamma/np.sqrt(1 + gamma**2)

@nb.jit("float64(float64, float64, float64, int64)")
def Am(freq, gamma, Ip, m):
    res = 0.0
    
    nu = Ip/freq*(1 + 1/(2*gamma**2))
    k0 = int(np.ceil(nu))
    
    p = 0.0
    
    for k in range(k0, k0+20):
        res += np.exp(-alpha(k - nu))*w_m(np.sqrt(betta(k - nu)), m)
    
    return res*(4/np.sqrt(3*np.pi))/factorial(m)*(gamma**2/(1 + gamma**2))

@nb.vectorize([nb.float64(nb.int64, nb.int64, nb.float64, nb.float64, nb.int64, nb.float64, nb.float64)])
def w_ppt(l, m, Cnl, Ip, Z, E, freq):
    if E < 0.000054:
        return 0.0
    
    k = np.sqrt(2*Ip)
    
    gamma = k*freq/E
    
    g = 3/(2*gamma)*((1 + 1/(2*gamma**2))*np.arcsinh(gamma) - np.sqrt(1 + gamma**2)/(2*gamma))
    
    F = E/k**3 # F/F0
    
    res = Ip * np.sqrt(3/(2*np.pi)) * Cnl**2 * (2*l + 1) * factorial(l + m) / (2**m * factorial(m)*factorial(l - m))
    res *= (2/(F*np.sqrt(1+gamma**2)))**(-m - 1.5 + 2*Z/k)
    res *= np.exp(-2/(3*F)*g)*Am(freq, gamma, Ip, m)
    
    return res
