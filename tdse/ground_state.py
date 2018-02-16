import numpy as np
from . import wavefunc, orbitals
from .grid import ShGrid


def wf_1(atom, l, m, grid, ws, dt, Nt):
    wf = wavefunc.SWavefunc.random(grid, l, m)

    for i in range(Nt):
        ws.prop_img(wf, atom, dt)
        wf.normalize()

    return wf

def wf_n(atom, n, l, m, grid, ws, dt, Nt):
    wfs = [wavefunc.SWavefunc.random(grid, l, m) for i in range(n)]

    for i in range(Nt):
        for j in range(n):
            wf = wfs[j]
            ws.prop_img(wf, atom, dt)
            wf.normalize()
        wavefunc.SWavefunc.ort_l(wfs, l)

    wfs[-1].normalize()
    return wfs[-1]

def wf(atom, grid, ws, dt, Nt, n=1, l=None, m=None):
    if l is None:
        l = atom.get_l(0)
    if m is None:
        m = atom.get_m(0)

    small_grid = ShGrid(grid.Nr, l+1, grid.Rmax)

    if n == 1:
        wf = wf_1(atom, l, m, small_grid, ws, dt, Nt)
    else:
        wf = wf_n(atom, n, l, m, small_grid, ws, dt, Nt)

    if small_grid.Nl == grid.Nl:
        wf_full = wf
    else:
        wf_full = wavefunc.SWavefunc(grid, m)
        wf_full.asarray()[l,:] = wf.asarray()[l,:]

    wf_full.asarray()[0:l,:] = 0.0
    wf_full.asarray()[l+1:,:] = 0.0

    return wf_full


def orbs(atom, grid, ws, dt=0.125, Nt=10000, print_calc_info=False):
    orbs = orbitals.SOrbitals(atom, grid)
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
