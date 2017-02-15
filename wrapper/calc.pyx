import numpy as np
cimport numpy as np

from wavefunc cimport SWavefunc
from workspace cimport SKnWorkspace
from field cimport Field
from hydrogen cimport hydrogen_sh_dudz


def az(SWavefunc wf, Field field, double t):
    return calc_az(wf.data, field.data, hydrogen_sh_dudz, t)

def az_t(int Nt, SKnWorkspace ws, SWavefunc wf, Field field):
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] az = np.ndarray(Nt, dtype=np.double)
    calc_az_t(Nt, &az[0], ws.data, wf.data, field.data)
    return az

def jrcd_t(SKnWorkspace ws, SWavefunc wf, Field field, int Nt, double t_smooth):
    return jrcd(ws.data, wf.data, field.data, hydrogen_sh_dudz, Nt, t_smooth)

def smstep(double x, double x0, double x1):
    return smoothstep(x, x0, x1)
