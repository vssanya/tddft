import numpy as np
cimport numpy as np

from atom cimport ShAtomCache
from wavefunc_gpu cimport ShWavefuncArrayGPU

def az(ShWavefuncArrayGPU wf_array, ShAtomCache atom_cache, double[:] E, np.ndarray[double, ndim=1, mode='c'] az = None):
    if az is None:
        az = np.ndarray(wf_array.N, dtype=np.double)
    calc_wf_array_gpu_az(wf_array.cdata[0], atom_cache.cdata[0], &E[0], &az[0])
    return az
