import numpy as np

def probability(n, j):
    res = 1.0
    for i in range(n.shape[0]):
        if i in j:
            res *= 1.0 - n[i]
        else:
            res *= n[i]
    return res

def n0(n):
    res = np.ndarray(n.shape[0])

    for i in range(n.shape[0]):
        res[i] = probability(n[i], [])

    return res

def n1(n):
    res = np.ndarray(n.shape[0])
    for i in range(n.shape[0]):
        res[i] = 0
        for j in range(n.shape[1]):
            res[i] += probability(n[i], [j])

    return res

def n2(n):
    res = np.ndarray(n.shape[0])
    for i in range(n.shape[0]):
        res[i] = 0
        for j1 in range(n.shape[1]-1):
            for j2 in range(j1+1, n.shape[1]):
                res[i] += probability(n[i], [j1, j2])
    return res

def n3(n):
    res = np.ndarray(n.shape[0])
    for i in range(n.shape[0]):
        res[i] = 0
        for j1 in range(n.shape[1]-2):
            for j2 in range(j1+1, n.shape[1]-1):
                for j3 in range(j2+1, n.shape[1]):
                    res[i] += probability(n[i], [j1,j2,j3])
    return res

def n4(n):
    res = np.ndarray(n.shape[0])
    for i in range(n.shape[0]):
        res[i] = 0
        for j1 in range(n.shape[1]-3):
            for j2 in range(j1+1, n.shape[1]-2):
                for j3 in range(j2+1, n.shape[1]-1):
                    for j4 in range(j3+1, n.shape[1]):
                        res[i] += probability(n[i], [j1,j2,j3,j4])
    return res
