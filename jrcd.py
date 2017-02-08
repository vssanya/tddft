from mpi4py import MPI

import numpy as np
from tdse import grid, wavefunc, field, workspace, hydrogen, calc, utils

def calc_jrcd(I0, length, t_fwhm, alpha, phase):
    freq = utils.length_to_freq(800, 'nm')
    E0 = utils.I_to_E(I0)
    tp = utils.t_fwhm(t_fwhm, 'fs')
    t0 = utils.t_shift(tp, I0, Imin=1e10)

    field = field.TwoColorPulseField(
        E0 = E0,
        alpha = alpha,
        freq = freq,
        phase = phase,
        tp = tp,
        t0 = t0
    )

    dt = 0.025
    dr = 0.1
    Nt = (2*t0)/dt

    r_max = utils.r_max(E, alpha, freq)

    grid = grid.SGrid(Nr=r_max/dr, Nl=40, r_max=r_max)
    wf = hydrogen.ground_state(grid)
    ws = workspace.SKnWorkspace(dt=dt, grid=grid)

    return calc.jrcd_t(ws, wf, field, Nt)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    phase = np.linspace(0.0, 2*np.pi, 20)

phase = comm.scatter(phase, root=0)

jrcd = calc_jrcd(
        I0=1e14,
        length=800,
        t_fwhm=50,
        alpha=0.2,
        phase=phase
        )

jrcd = comm.gather(jrcd, root=0)

if rank == 0:
    np.savetxt('res.txt', [phase, jrcd])
