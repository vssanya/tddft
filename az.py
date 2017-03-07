from mpi4py import MPI
import numpy as np

from tdse import grid, wavefunc, field, workspace, hydrogen, calc, utils

def calc_wf_az_t(tp, t0, rank):
    freq = utils.length_to_freq(800, 'nm')
    T = 2*np.pi/freq

    E0 = utils.I_to_E(1e14)

    tp = utils.t_fwhm(tp, 'fs')
    t0 = t0*T

    f = field.TwoColorPulseField(
        E0 = E0,
        alpha = 0.0,
        freq = freq,
        phase = 0.0,
        tp = tp,
        t0 = t0
    )

    dt = 0.025
    dr = 0.125
    r_max = 100

    g = grid.SGrid(Nr=r_max/dr, Nl=80, r_max=r_max)
    wf = hydrogen.ground_state(g)
    ws = workspace.SKnWorkspace(dt=dt, grid=g)

    t = np.arange(0, 2*t0, dt)
    az    = np.zeros(t.size)

    for it in range(t.size):
        #ws.prop(wf, f, t[it])
        az[it] = 1#calc.az(wf, f, t[it])

    np.savetxt('az_t_{}.txt'.format(rank), az)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

tp = [ 9.33, 3.99, 2.05 ]
t0 = [ 8, 3, 1.5 ]

calc_wf_az_t(tp[rank], t0[rank], rank)
