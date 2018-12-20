import numpy as np

from . import grid, wavefunc, orbitals
from .atom import AtomCache
from .grid import ShGrid


def wf_1(atom, grid, ws, dt, Nt, wf_class = wavefunc.ShWavefunc):
    l = atom.ground_state.l
    m = atom.ground_state.m
    wf = wf_class.random(grid, l, m)

    for i in range(Nt):
        wf.normalize()
        ws.prop_img(wf, dt)

    n = np.sqrt(wf.norm())
    E = 2/dt*(1-n)/(1+n)

    wf.normalize()

    return wf, E

def wf_n(atom, grid, ws, dt, Nt, wf_class = wavefunc.ShWavefunc):
    l = atom.ground_state.l
    m = atom.ground_state.m
    n = atom.ground_state.n + 1

    wfs = [wf_class.random(grid, l, m) for i in range(n)]

    for i in range(Nt):
        wf_class.ort_l(wfs, l)
        for j in range(n):
            wf = wfs[j]
            wf.normalize()
            ws.prop_img(wf, dt)

    n = np.sqrt(wfs[-1].norm())
    E = 2/dt*(1-n)/(1+n)

    wf_class.ort_l(wfs, l)
    wfs[-1].normalize()

    return wfs[-1], E

def wf(atom, grid, ws, dt, Nt, wf_class = wavefunc.ShWavefunc):
    l = atom.ground_state.l
    m = atom.ground_state.m
    n = atom.ground_state.n + 1

    small_grid = grid.createGridWith(l+1)

    if n == 1:
        wf, E = wf_1(atom, small_grid, ws, dt, Nt, wf_class)
    else:
        wf, E = wf_n(atom, small_grid, ws, dt, Nt, wf_class)

    if small_grid.Nl == grid.Nl:
        wf_full = wf
    else:
        wf_full = wf_class(grid, m)
        wf_full.asarray()[l,:] = wf.asarray()[l,:]

    wf_full.asarray()[0:l,:] = 0.0
    wf_full.asarray()[l+1:,:] = 0.0

    return wf_full, E


def orbs(atom, grid, ws, dt=0.125, Nt=10000, print_calc_info=False):
    orbs = orbitals.Orbitals(atom, grid)
    orbs.init()
    data = orbs.asarray()

    atom_cache = AtomCache(atom, grid)

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
