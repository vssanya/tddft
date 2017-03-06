from mpi4py import MPI

import numpy as np
import tdse


atom = tdse.atom.Atom('H')

def calc_jrcd(I0, length, t_fwhm, alpha, phase):
    freq = tdse.utils.length_to_freq(800, 'nm')
    E0 = tdse.utils.I_to_E(I0)
    tp = tdse.utils.t_fwhm(t_fwhm, 'fs')
    t0 = tdse.utils.t_shift(tp, I0, Imin=I0*1e-7)

    f = tdse.field.TwoColorPulseField(
        E0 = E0,
        alpha = alpha,
        freq = freq,
        phase = phase, tp = tp,
        t0 = t0
    )

    dt = 0.01
    dr = 0.1
    t_smooth = 200
    Nt = (2*t0 + t_smooth)/dt

    r_max = np.max([tdse.utils.r_max(E0, alpha, freq), 30])

    g = tdse.grid.SGrid(Nr=r_max/dr, Nl=40, r_max=r_max)
    wf = tdse.atom.ground_state(g)
    ws = tdse.workspace.SKnWorkspace(atom=atom, grid=g)

    tmp = np.array(wf.asarray())

    for i in range(50):
        ws.prop_img(wf, 0.1)
        wf.normalize()

    return tdse.calc.jrcd_t(ws, wf, f, Nt, dt, t_smooth)

def do_root(comm, count_workers):
    result = {}
    status = MPI.Status()

    phase = np.linspace(0.0, 2*np.pi, 20)
    I     = np.logspace(12, 15, 20)
    jrcd  = np.zeros((phase.size, I.size))

    ip = 0
    iI = 0

    work_status = 0
    finished_workers = 0

    while True:
        print("Status calculation = {} %".format(work_status/jrcd.size*100))

        res = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        if res['type'] == 'result':
            work_status += 1
            jrcd[res['index'][0],res['index'][1]] = res['jrcd']
        elif res['type'] == 'next_calc':
            if (ip+1)*(iI+1) >= jrcd.size:
                comm.send({
                    'type': 'finish'
                    }, dest=status.Get_source(), tag=1)
                finished_workers += 1
            else:
                comm.send({
                    'type': 'calc',
                    'param': {
                        'I0':     I[iI],
                        'length': 800,
                        't_fwhm': 50,
                        'alpha':  0.2,
                        'phase':  phase[ip]
                        },
                    'index': [ip, iI]
                    }, dest=status.Get_source(), tag=1)

                if ip == phase.size-1:
                    ip = 0
                    iI += 1
                else:
                    ip += 1

        if finished_workers >= count_workers:
            break

    np.savetxt('res_phase.txt', phase)
    np.savetxt('res_I.txt', I)
    np.savetxt('res_jrcd.txt', jrcd)

def do_worker(common):
    while True:
        comm.send({
            'type': 'next_calc'
            }, dest=0, tag=1)

        res = comm.recv(source=0, tag=1)

        if res['type'] == 'finish':
            break
        elif res['type'] == 'calc':
            jrcd = calc_jrcd(**res['param'])
            comm.send({
                'type': 'result',
                'index': res['index'],
                'jrcd': jrcd
                }, dest=0, tag=1)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    do_root(comm, size-1)
else:
    do_worker(comm)
