import numpy as np
from . import grid, wavefunc, field, workspace, hydrogen, calc, utils

def calc_hydrogen():
    dt = 0.001
    g = grid.ShGrid(Nr=1000, Nl=1, r_max=100)
    wf = wavefunc.SWavefunc.random(g)
    ws = workspace.SKnWorkspace(dt=dt, grid=g)

    prev_norm = 0.0
    norm = wf.norm()

    while np.abs(norm - prev_norm) / norm > 1e-5:
        ws.prop_img(wf)

        prev_norm = norm
        norm = wf.norm()
        print("Energy =", (np.sqrt(norm/prev_norm) - 1)/dt)

        wf.normalize()


    return wf
