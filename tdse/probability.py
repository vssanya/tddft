import numpy as np
from tdse.atom import Atom

def probability(n, j):
    res = 1.0
    for i in range(n.shape[0]):
        if i in j:
            res *= 1.0 - n[i]
        else:
            res *= n[i]
    return res

def transform_nsum(func):
    def new_func(atom: Atom, n):
        n_new = np.ndarray((n.shape[0], atom.countElectrons))
        for i in range(n.shape[0]):
            j = 0
            for orb in range(n.shape[1]):
                ne = atom.getCountElectrons(orb)
                for ni in range(ne):
                    n_new[i, j+ni] = n[i, orb] / ne
                j += ne

        return func(atom, n_new)

    return new_func

@transform_nsum
def n0(atom, n):
    res = np.ndarray(n.shape[0])

    for i in range(n.shape[0]):
        res[i] = probability(n[i], [])

    return res

@transform_nsum
def n1(atom, n):
    res = np.ndarray(n.shape[0])
    for i in range(n.shape[0]):
        res[i] = 0
        for j in range(n.shape[1]):
            res[i] += probability(n[i], [j])

    return res

@transform_nsum
def n2(atom, n):
    res = np.ndarray(n.shape[0])
    for i in range(n.shape[0]):
        res[i] = 0
        for j1 in range(n.shape[1]-1):
            for j2 in range(j1+1, n.shape[1]):
                res[i] += probability(n[i], [j1, j2])
    return res

@transform_nsum
def n3(atom, n):
    res = np.ndarray(n.shape[0])
    for i in range(n.shape[0]):
        res[i] = 0
        for j1 in range(n.shape[1]-2):
            for j2 in range(j1+1, n.shape[1]-1):
                for j3 in range(j2+1, n.shape[1]):
                    res[i] += probability(n[i], [j1,j2,j3])
    return res

@transform_nsum
def n4(atom, n):
    res = np.ndarray(n.shape[0])
    for i in range(n.shape[0]):
        res[i] = 0
        for j1 in range(n.shape[1]-3):
            for j2 in range(j1+1, n.shape[1]-2):
                for j3 in range(j2+1, n.shape[1]-1):
                    for j4 in range(j3+1, n.shape[1]):
                        res[i] += probability(n[i], [j1,j2,j3,j4])
    return res
