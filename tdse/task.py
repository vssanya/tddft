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

    def __init__(self, path_res='res', mode=None, save_state_step=None, send_status=True, term_save_state=True, is_mpi=False):
        """
        save_state_step is set in percents
        """

        self.res = {}
        self.mode = mode
        self.save_path = self._create_save_path(path_res)

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
        if self.is_slurm:
            self.bot_client = BotClient()

    def calc_init(self):
        if self.is_slurm:
            self.bot_client.start()

    def calc_prop(self, i, t):
        pass

    def calc_data(self, i, t):
        pass
        
    def calc(self, restart_index=0):
        self.calc_init()

        if restart_index != 0:
            self.load()
            self.load_state(restart_index)

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

        self.save()
        if self.is_slurm:
            self.bot_client.finish()

    def _create_save_path(self, path_res):
        script_path = inspect.getfile(self.__class__)
        task_dir = os.path.splitext(os.path.basename(script_path))[0]
        save_path = os.path.join(os.path.dirname(script_path), path_res, task_dir)
        if not os.path.exists(save_path):
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

class WavefuncTask(Task):
    """
    """
    state_n = 1 # Номер состояния

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

        self.grid = tdse.grid.ShGrid(Nr=self.r_max/self.dr, Nl=self.Nl, r_max=self.r_max)

    def save_state(self, i):
        np.save(os.path.join(self.save_path, 'wf_{}.npy'.format(i)), self.wf.asarray())

    def load_state(self, i):
        self.wf.asarray[:] = np.load(os.path.join(self.save_path, 'wf_{}.npy'.format(i)))

    def calc_init(self):
        super().calc_init()

        self.ws = tdse.workspace.SKnWorkspace(self.grid, self.uabs)
        self.wf = tdse.ground_state.wf(self.atom, self.grid, self.ws, self.dt, 10000, n=self.state_n)

        self.t = self.field.get_t(self.dt, nT=self.nT)

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.atom, self.field, t, self.dt)

class OrbitalsTask(Task):
    """
    """

    def __init__(self, path_res='res', mode=None, is_mpi=True, **kwargs):
        super().__init__(path_res, mode, is_mpi=is_mpi, **kwargs)

        self.sh_grid = tdse.grid.ShGrid(Nr=self.r_max/self.dr, Nl=self.Nl, r_max=self.r_max)
        self.sp_grid = tdse.grid.SpGrid(Nr=self.r_max/self.dr, Nc=Nc, Np=1, r_max=self.r_max)
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

        self.t = self.field.get_t(self.dt, nT=self.nT)

    def calc_prop(self, i, t):
        self.ws.prop(self.orbs, self.atom, self.field, t, self.dt)
