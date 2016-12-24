import time

import numpy as np
import matplotlib.pyplot as plt

from tdse import grid, wavefunc, field, workspace, hydrogen, calc


freq = 5.6e-2
E0 = 3.77e-2
T = 2.0*np.pi/freq
tp = T*2.05/2.67/(2*np.log(2))
dt = 0.025
Nt = 3*T/dt

t = np.linspace(0, 3*T, Nt)

field = field.TwoColorPulseField(
    E0 = E0,
    alpha = 0.0,
    freq = freq,
    phase = 0.0,
    tp = tp,
    t0 = 1.5*T
)

grid = grid.SGrid(Nr=2000, Nl=80, r_max=250)
wf = hydrogen.ground_state(grid)
ws = workspace.SKnWorkspace(dt=dt, grid=grid)

start_time = time.time()
a = calc.az_t(Nt, ws, wf, field)
print("Time of execution: {}".format(time.time() - start_time))

plt.plot(t, a)
plt.show()
