import os
import signal
import inspect
import numpy as np
from mpi4py import MPI

from .bot_client import BotClient

import tdse


class TaskError(Exception):
    pass


class Task(object):

    """
    Base class for task.
    Task represent calculation problem and contain all calculation parameters.
    Task save all results in res path in directory with name of main script.
    All calculated physics value must be listed in CALC_DATA = ['az', 'z'].
    Task can be used to load already calculated results for further analysis.
    """
    MODE_CALC = 1
    MODE_ANALISIS = 2

    CALC_DATA = []

    save_path = None

    def __init__(self, path_res='res', mode=None, save_inter_data=True, save_state_step=None, send_status=True, term_save_state=True, is_mpi=False):
        """
        save_state_step is set in percents
        """

        self.res = {}
        self.mode = mode

        self.signal_term = False

        self.save_inter_data = save_inter_data
        self.save_state_step = save_state_step
        self.send_status = send_status
        self.term_save_state = term_save_state

        if mode is None:
            mode = Task.MODE_CALC

        if mode is Task.MODE_CALC:
            def handler(signal, frame):
                self.signal_term = True
            signal.signal(signal.SIGTERM, handler)
        else:
            is_mpi = False

        self.is_mpi = is_mpi
        if self.is_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_rank()
        else:
            self.comm = None
            self.rank = 0
            self.szie = 1

        self.is_slurm = os.environ.get('SLURM_JOB_ID', None) is not None
        if mode is Task.MODE_ANALISIS:
            self.send_status = False
        if self.is_slurm and self.send_status and self.is_root():
            self.bot_client = BotClient()
        else:
            self.bot_client = None

        self.save_path = self._create_save_path(path_res)

    def finish_with_error(self, message):
        if self.bot_client:
            self.bot_client.send_message("Error: {}".format(message))

        raise TaskError()

    def is_root(self):
        return (not self.is_mpi) or (self.rank == 0)

    def calc_init(self):
        if self.bot_client is not None:
            self.bot_client.start()

    def calc_prop(self, i, t):
        pass

    def calc_data(self, i, t):
        pass

    def calc_finish(self):
        pass

    def calc(self, restart_index=0):
        if self.bot_client is not None:
            self.bot_client.send_message("Start calc")

        self.calc_init()

        if restart_index != 0:
            self.load()
            self.load_state(restart_index)

        print("Start propogation")
        for i in range(restart_index, self.t.size):
            self.calc_prop(i, self.t[i])
            self.calc_data(i, self.t[i])

            if (self.save_inter_data or self.bot_client is not None) and (i+1) % int(self.t.size*0.01) == 0:
                if self.save_inter_data:
                    self.save()

                if self.bot_client is not None:
                    self.bot_client.send_status(i / self.t.size)

            if self.save_state_step is not None and (i+1) % int(self.t.size*self.save_state_step/100) == 0:
                self.save_state(i)

            if self.signal_term:
                if self.term_save_state:
                    self.save_state(i)
                break

        self.calc_finish()

        self.save()
        if self.bot_client is not None:
            self.bot_client.finish()

    def _create_save_path(self, path_res):
        if self.save_path is None:
            script_path = inspect.getfile(self.__class__)
            task_dir = os.path.splitext(os.path.basename(script_path))[0]
            save_path = os.path.join(os.path.dirname(script_path), path_res, task_dir)
        else:
            save_path = self.save_path

        if self.rank == 0 and not os.path.exists(save_path):
            os.mkdir(save_path)

        return save_path

    def _get_data_path(self, calc_data_name):
        return os.path.join(self.save_path, "{}.npy".format(calc_data_name))

    def save(self):
        if self.rank == 0:
            for name in self.CALC_DATA:
                np.save(self._get_data_path(name), getattr(self, name))

    def save_state(self, i):
        pass

    def load_state(self):
        pass

    def load(self):
        if self.rank == 0:
            for name in self.CALC_DATA:
                setattr(self, name, np.load(self._get_data_path(name)))


class TaskAtom(Task):
    atom = tdse.atom.H
    atom_u_data_path = None

    dr = 0.025
    r_max = 100
    Nl = 2

    uabs = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Nr = int(self.r_max/self.dr)
        self.sh_grid = tdse.grid.ShGrid(Nr=self.Nr, Nl=self.Nl, r_max=self.r_max)

        if self.atom_u_data_path is None:
            atom_u_data = None
        else:
            atom_u_data = np.load(self.atom_u_data_path)

        self.atom_cache = tdse.atom.AtomCache(self.atom, self.sh_grid, atom_u_data)

        if self.uabs is not None:
            self.uabs_cache = tdse.abs_pot.UabsCache(self.uabs, self.sh_grid)


class SFATask(Task):
    Nr = 100
    Nc = 50

    Rmax = 4

    field = None

    dt = 0.025

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)
        self.grid = tdse.grid.SpGrid2d(self.Nr, self.Nc, self.Rmax)

    def calc_init(self):
        self.ws = tdse.workspace.SFAWorkspace()
        self.wf = tdse.wavefunc.CtWavefunc(self.grid)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.field, t, self.dt)

class WavefuncTask(TaskAtom):
    """
    """
    is_calc_ground_state = True

    Workspace = tdse.workspace.SKnWorkspace

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

    def save_state(self, i):
        np.save(os.path.join(self.save_path, 'wf_{}.npy'.format(i)), self.wf.asarray())

    def load_state(self, i):
        self.wf.asarray[:] = np.load(os.path.join(self.save_path, 'wf_{}.npy'.format(i)))

    def calc_init(self):
        super().calc_init()

        self.ws = self.Workspace(self.atom_cache, self.sh_grid, self.uabs_cache)

        if self.is_calc_ground_state:
            print("Start calc ground state")
            self.wf = self.calc_ground_state(self.ws)
        else:
            self.wf = tdse.wavefunc.ShWavefunc(self.sh_grid)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_ground_state(self, ws=None):
        if ws is None:
            ws = self.Workspace(self.atom_cache, self.sh_grid, tdse.abs_pot.UabsZero())

        return tdse.ground_state.wf(self.atom, self.sh_grid, ws, self.dt, 10000)[0]

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.field, t, self.dt)

class WfGroundStateTask(TaskAtom):
    atom = tdse.atom.H

    dt = 0.025
    dr = 0.125

    T = 100
    r_max = 100

    CALC_DATA = ['wf_gs', 'E']

    def __init__(self, path_res='res', mode=None, is_mpi=False, **kwargs):
        self.Nl = self.atom.l_max + 1
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

        self.Nt = int(self.T / self.dt)

    def calc_init(self):
        super().calc_init()

        self.ws = tdse.workspace.SKnWorkspace(self.atom_cache, self.sh_grid, tdse.abs_pot.UabsZero())

    def calc(self):
        self.calc_init()

        self.wf, self.E = tdse.ground_state.wf(self.atom, self.sh_grid, self.ws, self.dt, self.Nt, **self.state)

        self.wf_gs = self.wf.asarray()
        self.save()

class WavefuncWithSourceTask(TaskAtom):
    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

        self.grid_source = tdse.grid.ShGrid(Nr=self.sh_grid.Nr, Nl=self.atom.l_max+1, r_max=self.r_max)

    def save_state(self, i):
        np.save(os.path.join(self.save_path, 'wf_{}.npy'.format(i)), self.wf.asarray())

    def load_state(self, i):
        self.wf.asarray[:] = np.load(os.path.join(self.save_path, 'wf_{}.npy'.format(i)))

    def calc_init(self):
        super().calc_init()

        self.wf_source, self.Ip = self.calc_ground_state()
        self.wf = tdse.wavefunc.ShWavefunc(self.sh_grid)
        self.wf.asarray()[:] = 0.0

        self.ws = tdse.workspace.SKnWithSourceWorkspace(tdse.atom.AtomCache(tdse.atom.NONE, self.sh_grid), self.sh_grid, self.uabs_cache, self.wf_source, self.Ip)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_ground_state(self, ws=None):
        if ws is None:
            ws = tdse.workspace.SKnWorkspace(self.atom_cache, self.grid_source, tdse.abs_pot.UabsCache(tdse.abs_pot.UabsZero(), self.grid_source))

        return tdse.ground_state.wf(self.atom, self.grid_source, ws, self.dt, 10000)

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.field, t, self.dt)

class OrbitalsTask(TaskAtom):
    atom = tdse.atom.Ar

    dt = 0.008
    dr = 0.025

    r_max = 100
    Nl = 2
    Nc = 33

    uxc = tdse.hartree_potential.UXC_LB
    Uxc_lmax = 1
    Uh_lmax = 3

    ground_state = None
    ground_state_task = None

    def __init__(self, path_res='res', mode=None, is_mpi=True, **kwargs):
        if self.ground_state_task is not None:
            self.dt    = self.ground_state_task.dt
            self.dr    = self.ground_state_task.dr
            self.r_max = self.ground_state_task.r_max
            self.uxc   = self.ground_state_task.uxc
            self.Nc    = self.ground_state_task.Nc
            self.atom  = self.ground_state_task.atom

        super().__init__(path_res, mode, is_mpi=is_mpi, **kwargs)

        self.sp_grid = tdse.grid.SpGrid(Nr=self.r_max/self.dr, Nc=self.Nc, Np=1, r_max=self.r_max)
        self.ylm_cache = tdse.sphere_harmonics.YlmCache(self.Nl, self.sp_grid)

    def _get_state_filename(self, i):
        return os.path.join(self.save_path, 'orbs_{}.npy'.format(i))

    def save_state(self, i):
        self.orbs.save(self._get_state_filename(i))

    def load_state(self, i):
        self.orbs.load(self._get_state_filename(i))

    def calc_init(self):
        super().calc_init()

        self.orbs = tdse.orbitals.Orbitals(self.atom, self.sh_grid, self.comm)
        self.orbs.load(self.ground_state)

        self.ws = tdse.workspace.SOrbsWorkspace(self.atom_cache, self.sh_grid, self.sp_grid, self.uabs_cache, self.ylm_cache, Uxc_lmax=self.Uxc_lmax, Uh_lmax = self.Uh_lmax, uxc=self.uxc)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_prop(self, i, t):
        self.ws.prop(self.orbs, self.field, t, self.dt)

class OrbitalsGroundStateTask(TaskAtom):
    atom = tdse.atom.Ar

    dt = 0.008
    dr = 0.025

    T = 100
    r_max = 100
    Nc = 33

    uxc = tdse.hartree_potential.UXC_LB
    uabs = tdse.abs_pot.UabsZero()

    Uxc_lmax = 1
    Uh_lmax = 1

    CALC_DATA = ['orbs_gs', 'E', 'uee']

    def __init__(self, path_res='res', mode=None, is_mpi=False, **kwargs):
        self.Nl = self.atom.l_max + 1

        super().__init__(path_res, mode, is_mpi=False, **kwargs)

        self.Nt = int(self.T / self.dt)

        self.sp_grid = tdse.grid.SpGrid(Nr=self.Nr, Nc=self.Nc, Np=1, r_max=self.r_max)
        self.ylm_cache = tdse.sphere_harmonics.YlmCache(self.Nl, self.sp_grid)

    def calc_init(self):
        super().calc_init()

        self.ws = tdse.workspace.SOrbsWorkspace(self.atom_cache, self.sh_grid, self.sp_grid, self.uabs_cache, self.ylm_cache, Uxc_lmax=self.Uxc_lmax, Uh_lmax = self.Uh_lmax, uxc=self.uxc)

    def calc(self):
        self.calc_init()

        self.orbs, self.E = tdse.ground_state.orbs(self.atom, self.sh_grid, self.ws, self.dt, self.Nt, print_calc_info=True)
        self.orbs_gs = self.orbs.asarray()

        self.ws.calc_uee(self.orbs)
        self.uee = self.ws.uee[0]

        self.save()

class TdsfmInnerTask(Task):
    CALC_DATA = ['p',]

    sp_grid = None
    l_min = 0
    l_max = -1

    without_ground_state = False

    def __init__(self, tdsfm_task, **kwargs):
        super().__init__(is_mpi=False, **kwargs)

        self.task = tdsfm_task

    def calc_init(self):
        self.task.calc_init()
        self.task.load()
        self.task.wf.asarray()[:] = self.task.psi[:]

        if self.sp_grid is None:
            self.tdsfm = self.task.tdsfm
        else:
            self.tdsfm = tdse.tdsfm.TDSFM_VG(self.sp_grid, self.task.sh_grid, 0)

        self.p = self.tdsfm.asarray()

    def calc(self):
        self.calc_init()

        if (self.without_ground_state):
            self.task.wf.asarray()[:] = self.task.wf.asarray()[:] - (self.task.wf*self.task.wf_gs)*self.task.wf_gs.asarray()[:]

        self.tdsfm.calc_inner(self.task.field, self.task.wf, 0, 0, self.task.sh_grid.Nr, self.l_min, self.l_max)

        self.save()
