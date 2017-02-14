import numpy as np
import matplotlib.pyplot as plt

from tdse import grid, wavefunc, field, workspace, hydrogen, calc, utils

freq = utils.length_to_freq(800, 'nm')
T = 2*np.pi/freq

E0 = utils.I_to_E(1e14)

tp = utils.t_fwhm(9.33, 'fs')
t0 = 8*T

dt = 0.025
dr = 0.125
r_max = 100

g = grid.SGrid(Nr=r_max/dr, Nl=80, r_max=r_max)
wf = hydrogen.ground_state(g)
ws = workspace.SKnWorkspace(dt=dt, grid=g)
