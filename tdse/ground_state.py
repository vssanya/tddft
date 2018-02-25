import numpy as np
from . import grid, wavefunc, orbitals


def wf_1(atom, l, m, grid, ws, dt, Nt):
    wf = wavefunc.ShWavefunc.random(grid, l, m)

    for i in range(Nt):
        ws.prop_img(wf, atom, dt)
        wf.normalize()

    return wf

def wf_n(atom, n, l, m, grid, ws, dt, Nt):
    wfs = [wavefunc.ShWavefunc.random(grid, l, m) for i in range(n)]

    for i in range(Nt):
        for j in range(n):
            ws.prop_img(wfs[j], atom, dt)
            wfs[j].normalize()
        wavefunc.ShWavefunc.ort_l(wfs, l)

    wfs[-1].normalize()
    return wfs[-1]

def wf(atom, grid, ws, dt, Nt, n=1, l=None, m=None):
    if l is None:
        l = atom.get_l(0)
    if m is None:
        m = atom.get_m(0)

    if n == 1:
        return wf_1(atom, l, m, grid, ws, dt, Nt)
    else:
        return wf_n(atom, n, l, m, grid, ws, dt, Nt)


def orbs(atom, grid, ws, dt=0.125, Nt=10000, print_calc_info=False):
    orbs = orbitals.Orbitals(atom, grid)
    orbs.init()

    for i in range(Nt):
        if print_calc_info and i % (Nt // 100) == 0:
            print("i = ", i//100)
            n = np.sqrt(orbs.norm_ne())
            print(2/dt*(1-n)/(1+n))

        orbs.ort()
        orbs.normalize()
        ws.prop_img(orbs, atom, dt)

    n = np.sqrt(orbs.norm_ne())
    E = 2/dt*(1-n)/(1+n)

    orbs.ort()
    orbs.normalize()

    return orbs, E
