import numpy as np
from . import grid, wavefunc, orbitals
from .atom import AtomCache


def wf_1(atom, grid, ws, dt, Nt):
    l = atom.ground_state.l
    m = atom.ground_state.m
    wf = wavefunc.ShWavefunc.random(grid, l, m)

    for i in range(Nt):
        ws.prop_img(wf, dt)
        wf.normalize()

    return wf

def wf_n(atom, grid, ws, dt, Nt):
    l = atom.ground_state.l
    m = atom.ground_state.m
    n = atom.ground_state.n

    wfs = [wavefunc.ShWavefunc.random(grid, l, m) for i in range(n)]

    for i in range(Nt):
        for j in range(n):
            ws.prop_img(wfs[j], dt)
            wfs[j].normalize()
        wavefunc.ShWavefunc.ort_l(wfs, l)

    wfs[-1].normalize()
    return wfs[-1]

def wf(atom, grid, ws, dt, Nt):
    if atom.ground_state.n == 1:
        return wf_1(atom, grid, ws, dt, Nt)
    else:
        return wf_n(atom, grid, ws, dt, Nt)


def orbs(atom, grid, ws, dt=0.125, Nt=10000, print_calc_info=False):
    orbs = orbitals.Orbitals(atom, grid)
    orbs.init()
    data = orbs.asarray()

    atom_cache = AtomCache(atom, grid)
    print(atom_cache.get_u())

    for i in range(Nt):
        if print_calc_info and i % (Nt // 100) == 0:
            print("i = ", i//100)
            n = np.sqrt(orbs.norm_ne())
            print(2/dt*(1-n)/(1+n))

        orbs.ort()
        orbs.normalize()
        ws.prop_img(orbs, dt)

    n = np.sqrt(orbs.norm_ne())
    E = 2/dt*(1-n)/(1+n)

    orbs.ort()
    orbs.normalize()

    return orbs, E
