import numpy as np

from . import grid, wavefunc, orbitals
from .grid import ShGrid


def wf_1(atom, grid, ws, dt, Nt, wf_class = wavefunc.ShWavefunc):
    l = atom.ground_state.l
    m = atom.ground_state.m
    wf = wf_class.random(grid, l, m)

    Elast = 0.0
    E = 1.0

    while np.abs(Elast - E) > 1e-6:
        for i in range(Nt):
            wf.normalize()
            ws.prop_img(wf, dt)

        n = np.sqrt(wf.norm())

        Elast = E
        E = 2/dt*(1-n)/(1+n)

    wf.normalize()

    return wf, E

def wf_n(atom, grid, ws, dt, Nt, wf_class = wavefunc.ShWavefunc):
    l = atom.ground_state.l
    m = atom.ground_state.m
    n = atom.ground_state.n + 1

    wfs = [wf_class.random(grid, l, m) for i in range(n)]

    Elast = 0.0
    E = 1.0

    while np.abs(Elast - E) > 1e-6:
        for i in range(Nt):
            wf_class.ort_l(wfs, l)
            for j in range(n):
                wf = wfs[j]
                wf.normalize()
                ws.prop_img(wf, dt)

        norm = np.sqrt(wfs[-1].norm())
        Elast = E
        E = 2/dt*(1-norm)/(1+norm)

    n = np.sqrt(wfs[-1].norm())
    E = 2/dt*(1-n)/(1+n)

    wf_class.ort_l(wfs, l)
    wfs[-1].normalize()

    return wfs[-1], E

def wf(atom, grid, ws, dt, Nt, wf_class = wavefunc.ShWavefunc, ground_state = None):
    if ground_state is None:
        ground_state = atom.ground_state

    l = ground_state.l
    m = ground_state.m
    n = ground_state.n + 1

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


def orbs(atom, grid, ws, dt, Nt, orbitals_class, atom_cache_class,
        print_calc_info=False, dt_count=None, norm=None):
    orbs = orbitals_class(atom, grid)
    orbs.init()

    if norm == None:
        orb_norm = np.ones_like(atom.orbCountElectrons)*atom.orbCountElectrons
    else:
        orb_norm = norm*atom.orbCountElectrons

    for i in range(Nt):
        if print_calc_info and i % (Nt // 100) == 0:
            print("i = ", i//100)
            n = np.sqrt(orbs.norm_ne() / orb_norm)
            print(2/dt*(1-n)/(1+n))

        orbs.ort()
        orbs.normalize(norm)
        ws.prop_img(orbs, dt, dt_count=dt_count)

    n = np.sqrt(orbs.norm_ne() / orb_norm)
    E = 2/dt*(1-n)/(1+n)

    orbs.ort()
    orbs.normalize()

    return orbs, E

def orbs_two_step_uee(atom, grid, ws, dt, Nt, orbitals_class, atom_cache_class, print_calc_info=False, dt_count=None):
    orbs = orbitals_class(atom, grid)
    orbs.init()
    orbs_data = orbs.asarray()

    uxc_prev = np.copy(ws.uee)
    uxc_next = np.copy(ws.uee)
    orbs_prev = np.copy(orbs_data)

    atom_cache = atom_cache_class(atom, grid)

    # Nt1 = Nt//10
    # for i in range(Nt1):
        # if print_calc_info and i % (Nt1 // 100) == 0:
            # print("i = ", i//100)
            # n = np.sqrt(orbs.norm_ne() / atom.orbCountElectrons)
            # print(2/dt*(1-n)/(1+n))

        # orbs.ort()
        # orbs.normalize()

        # orbs_data[6,:] = orbs_data[4,:]
        # orbs_data[5,:] = orbs_data[3,:]

        # ws.prop_img(orbs, dt, dt_count=dt_count)

    dt = dt / 10

    for i in range(Nt):
        if print_calc_info and i % (Nt // 100) == 0:
            print("i = ", i//100)
            n = np.sqrt(orbs.norm_ne() / atom.orbCountElectrons)
            print(2/dt*(1-n)/(1+n))

        orbs.ort()
        orbs.normalize()

        orbs_data[6,:] = orbs_data[4,:]
        orbs_data[5,:] = orbs_data[3,:]

        ws.calc_uee(orbs)
        ws.uee[:] = (2*ws.uee[:] - uxc_prev[:])

        orbs_prev[:] = orbs_data[:]
        uxc_prev[:] = ws.uee[:]

        err = 2

        while err > 1e-1:
            ws.prop_img(orbs, dt, dt_count=dt_count, calc_uee=False)

            orbs_data[:] = 0.5*(orbs_data[:] + orbs_prev[:])
            ws.calc_uee(orbs, 1, 1)

            err = np.sum(np.abs(ws.uee[0,:300] - uxc_prev[0,:300]))

            if print_calc_info and i % (Nt // 100) == 0:
                print(err)
                print("Arg max = ", np.argmax(np.abs(ws.uee)), ", Max = ", np.max(np.abs(ws.uee)))

            orbs_data[:] = orbs_prev[:]
            uxc_prev[:] = ws.uee[:]

        ws.prop_img(orbs, dt, dt_count=dt_count, calc_uee=False)


    n = np.sqrt(orbs.norm_ne() / atom.orbCountElectrons)
    E = 2/dt*(1-n)/(1+n)

    orbs.ort()
    orbs.normalize()

    return orbs, E

def orbs_step_shells(atom, grid, ws, dt, Nt, orbitals_class, atom_cache_class, print_calc_info=False, dt_count=None, norm = None):
    orbs = orbitals_class(atom, grid)
    data = orbs.asarray()
    data[:] = 0.0

    if norm == None:
        orb_norm = np.ones_like(atom.orbCountElectrons)*atom.orbCountElectrons
    else:
        orb_norm = norm*atom.orbCountElectrons

    for shell in range(atom.countShells):
        orbs.init_shell(shell)

        active_orbs = atom.getActiveOrbs(shell)

        Esum = 0.0
        Esum_last = 0.0

        for i in range(Nt):
            if print_calc_info and i % (Nt // 100) == 0:
                n = np.sqrt(orbs.norm_ne() / orb_norm)
                E = 2/dt*(1-n)/(1+n)

                Esum_last = Esum
                Esum = np.sum(E[active_orbs])

                print("i = ", i//100)
                print(E)
                print("E_sum = {}".format(Esum))

                if np.abs(Esum - Esum_last) < 1e-6:
                    break

            orbs.ort()
            orbs.normalize(active_orbs, norm)
            ws.prop_img(orbs, dt, active_orbs, dt_count=dt_count)

    n = np.sqrt(orbs.norm_ne() / orb_norm)
    E = 2/dt*(1-n)/(1+n)

    orbs.ort()
    orbs.normalize()

    return orbs, E
