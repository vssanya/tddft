import numpy as np

from . import grid, wavefunc, orbitals
from .atom import AtomCache
from .grid import ShGrid


def wf_1(atom, grid, ws, dt, Nt):
    l = atom.ground_state.l
    m = atom.ground_state.m
    wf = wavefunc.ShWavefunc.random(grid, l, m)

    for i in range(Nt):
        wf.normalize()
        ws.prop_img(wf, dt)

    n = np.sqrt(wf.norm())
    E = 2/dt*(1-n)/(1+n)

    wf.normalize()

    return wf, E

def wf_n(atom, grid, ws, dt, Nt):
    l = atom.ground_state.l
    m = atom.ground_state.m
    n = atom.ground_state.n + 1

    wfs = [wavefunc.ShWavefunc.random(grid, l, m) for i in range(n)]

    for i in range(Nt):
        wavefunc.ShWavefunc.ort_l(wfs, l)
        for j in range(n):
            wf = wfs[j]
            wf.normalize()
            ws.prop_img(wf, dt)

    n = np.sqrt(wfs[-1].norm())
    E = 2/dt*(1-n)/(1+n)

    wavefunc.ShWavefunc.ort_l(wfs, l)
    wfs[-1].normalize()

    return wfs[-1], E

def wf(atom, grid, ws, dt, Nt):
    l = atom.ground_state.l
    m = atom.ground_state.m
    n = atom.ground_state.n + 1

    small_grid = ShGrid(grid.Nr, l+1, grid.Rmax)

    if n == 1:
        wf, E = wf_1(atom, small_grid, ws, dt, Nt)
    else:
        wf, E = wf_n(atom, small_grid, ws, dt, Nt)

    if small_grid.Nl == grid.Nl:
        wf_full = wf
    else:
        wf_full = wavefunc.ShWavefunc(grid, m)
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
