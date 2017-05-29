import numpy as np
import matplotlib.pyplot as plt

import tdse


freq = tdse.utils.length_to_freq(800, 'nm')
T = 2*np.pi/freq

I = 1e14
E0 = tdse.utils.I_to_E(I)

tp = tdse.utils.t_fwhm(2.05, 'fs')
t0 = 1.5*T

#f = tdse.field.TwoColorPulseField(E0=E0, freq=freq, alpha=0.0, tp=tp, t0=t0)
f = tdse.field.TwoColorTrField(E0=E0, freq=freq, alpha=0.0, tp=tp, t0=tp*0.1)

t = np.arange(0, tp, 0.008)
E = np.zeros(t.size)

for i in range(t.size):
    E[i] = f.E(t[i])

print(t)
print(E)

plt.plot(t, E)
plt.show()
