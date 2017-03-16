import numpy as np
cimport numpy as np


def djdt(np.ndarray[double] E, double freq, double alpha, double phase):
    cdef int i, n = E.shape[0]
    cdef np.ndarray[double] res = np.ndarray(n)

    for i in range(n):
        res[i] = sfa_djdt(E[i], freq, alpha, phase)

    return res

def dwdt(np.ndarray[double] E, double freq, double alpha, double phase):
    cdef int i, n = E.shape[0]
    cdef np.ndarray[double] res = np.ndarray(n)

    for i in range(n):
        res[i] = sfa_dwdt(E[i], freq, alpha, phase)

    return res
