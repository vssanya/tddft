import os
import signal
import inspect
import numpy as np
from mpi4py import MPI

from .bot_client import BotClient

import tdse


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

    def __init__(self, path_res='res', mode=None, save_state_step=None, send_status=True, term_save_state=True, is_mpi=False):
        """
        save_state_step is set in percents
        """

        self.res = {}
        self.mode = mode

        self.signal_term = False

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
        if self.is_slurm and self.send_status:
            self.bot_client = BotClient()

        self.save_path = self._create_save_path(path_res)

    def calc_init(self):
        if self.is_slurm and self.send_status:
            self.bot_client.start()

    def calc_prop(self, i, t):
        pass

    def calc_data(self, i, t):
        pass

    def calc_finish(self):
        pass
        
    def calc(self, restart_index=0):
        self.calc_init()

        if restart_index != 0:
            self.load()
            self.load_state(restart_index)

        print("Start propogation")
        for i in range(restart_index, self.t.size):
            self.calc_prop(i, self.t[i])
            self.calc_data(i, self.t[i])

            if (i+1) % int(self.t.size*0.01) == 0:
                self.save()

                if self.is_slurm and self.send_status:
                    self.bot_client.send_status(i / self.t.size)

            if self.save_state_step is not None and (i+1) % int(self.t.size*self.save_state_step/100) == 0:
                self.save_state(i)

            if self.signal_term:
                if self.term_save_state:
                    self.save_state(i)
                break

        self.calc_finish()

        self.save()
        if self.is_slurm and self.send_status:
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

class SFATask(Task):
    Nr = 100
    Nc = 50

    Rmax = 4

    field = None

    dt = 0.025

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)
        self.grid = tdse.grid.Sp2Grid(self.Nr, self.Nc, self.Rmax)

    def calc_init(self):
        self.ws = tdse.workspace.SFAWorkspace()
        self.wf = tdse.wavefunc.CtWavefunc(self.grid)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.field, t, self.dt)

class WavefuncTask(Task):
    """
    """
    state = {'n': 1, 'l': 0, 'm': 0}
    is_calc_ground_state = True

    Workspace = tdse.workspace.SKnWorkspace

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

        self.grid = tdse.grid.ShGrid(Nr=self.r_max/self.dr, Nl=self.Nl, r_max=self.r_max)

    def save_state(self, i):
        np.save(os.path.join(self.save_path, 'wf_{}.npy'.format(i)), self.wf.asarray())

    def load_state(self, i):
        self.wf.asarray[:] = np.load(os.path.join(self.save_path, 'wf_{}.npy'.format(i)))

    def calc_init(self):
        super().calc_init()

        self.ws = self.Workspace(self.grid, self.uabs)

        if self.is_calc_ground_state:
            print("Start calc ground state")
            self.wf = self.calc_ground_state(self.ws)
        else:
            self.wf = tdse.wavefunc.SWavefunc(self.grid)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_ground_state(self, ws=None):
        if ws is None:
            ws = self.Workspace(self.grid, tdse.abs_pot.UabsZero())

        return tdse.ground_state.wf(self.atom, self.grid, ws, self.dt, 10000, **self.state)

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.atom, self.field, t, self.dt)

class WavefuncWithSourceTask(Task):
    state = {'n': 1, 'l': 0, 'm': 0}

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

        Nr = int(self.r_max/self.dr)
        self.grid = tdse.grid.ShGrid(Nr=Nr, Nl=self.Nl, r_max=self.r_max)
        self.grid_source = tdse.grid.ShGrid(Nr=Nr, Nl=self.state['l']+1, r_max=self.r_max)

    def save_state(self, i):
        np.save(os.path.join(self.save_path, 'wf_{}.npy'.format(i)), self.wf.asarray())

    def load_state(self, i):
        self.wf.asarray[:] = np.load(os.path.join(self.save_path, 'wf_{}.npy'.format(i)))

    def calc_init(self):
        super().calc_init()

        self.wf_source = self.calc_ground_state()
        self.wf = tdse.wavefunc.SWavefunc(self.grid)
        self.wf.asarray()[:] = 0.0

        self.ws = tdse.workspace.SKnWithSourceWorkspace(self.grid, self.uabs, self.wf_source, -0.5)


        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_ground_state(self, ws=None):
        if ws is None:
            ws = tdse.workspace.SKnWorkspace(self.grid_source, tdse.abs_pot.UabsZero())

        return tdse.ground_state.wf(self.atom, self.grid, ws, self.dt, 10000, **self.state)

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, tdse.atom.NONE, self.field, t, self.dt)

class OrbitalsTask(Task):
    """
    """

    def __init__(self, path_res='res', mode=None, is_mpi=True, **kwargs):
        super().__init__(path_res, mode, is_mpi=is_mpi, **kwargs)

        self.sh_grid = tdse.grid.ShGrid(Nr=self.r_max/self.dr, Nl=self.Nl, r_max=self.r_max)
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

        self.orbs = tdse.orbitals.SOrbitals(self.atom, self.sh_grid, self.comm)
        self.orbs.load(self.ground_state)

        self.ws = tdse.workspace.SOrbsWorkspace(self.sh_grid, self.sp_grid, self.uabs, self.ylm_cache, Uxc_lmax=self.Uxc_lmax, Uh_lmax = self.Uh_lmax, uxc=self.uxc)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_prop(self, i, t):
        self.ws.prop(self.orbs, self.atom, self.field, t, self.dt)
