import os
import signal
import inspect
import numpy as np
import h5py

from mpi4py import MPI

from ..bot_client import BotClient

import tdse
from tdse.ui.progressbar import ProgressBar


class CalcData(object):
    NAME = "name"
    DTYPE = np.double

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", self.NAME)
        self.dtype = kwargs.get("dtype", self.DTYPE)

    def get_shape(self, task):
        pass

    def calc_init(self, task, file: h5py.File):
        self.file = file

        if task.rank is None or task.rank == 0:
            self.dset = file.create_dataset(self.name,
                                            shape=self.get_shape(task),
                                            dtype=self.dtype)

    def calc(self, task, i, t):
        pass

    def load(self, task, file: h5py.File):
        setattr(task, self.name, file[self.name][:])


class CalcDataWithMask(CalcData):
    def __init__(self, mask=None, **kwargs):
        super().__init__(**kwargs)

        self.mask = mask

    def calc_init(self, task, file):
        super().calc_init(task, file)

        if self.mask is not None:
            self.mask.write_params(self.dset)


class TimeShapeMixin(object):
    def get_shape(self, task):
        return (task.t.size,)


class TaskError(Exception):
    pass


class Task(object):

    """
    Base class for task.
    Task represent calculation problem and contain all calculation parameters.
    Task save all results in res path in directory with name of main script.
    All calculated physics value must be listed in CALC_DATA = [AzOrb(), NOrb()].
    Task can be used to load already calculated results for further analysis.
    """
    MODE_CALC = 1
    MODE_ANALISIS = 2

    CALC_DATA = [] # Массив объектов унаследованных от класса CalcData

    save_path = None

    def __init__(self, path_res='res', mode=None, save_inter_data=True, save_state_step=None, send_status=True, term_save_state=True, is_mpi=False):
        """
        save_state_step is set in percents
        """

        self.res = {}
        self.mode = mode

        self.file = None # hdf5 calculated data

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

        if self.rank == 0:
            self.file = h5py.File("{}.hdf5".format(self.save_path), 'w')
            self.write_calc_params(self.file.create_group("params"))

    def data_init(self):
        for data in self.CALC_DATA:
            if type(data) is not str:
                data.calc_init(self, self.file)

    def write_calc_params(self, params_grp: h5py.Group):
        pass

    def calc_prop(self, i, t):
        pass

    def calc_data(self, i, t):
        for data in self.CALC_DATA:
            if type(data) is not str:
                data.calc(self, i, t)

    def calc_finish(self):
        pass

    def calc(self, restart_index=0):
        if self.bot_client is not None:
            self.bot_client.send_message("Start calc")

        self.calc_init()
        self.data_init()

        if restart_index != 0:
            self.load()
            self.load_state(restart_index)

        progressBar = ProgressBar(self.t.size, prefix="Progress of propagation")
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

            if self.is_root() and i%100 == 0:
                progressBar.print(i)

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
        if self.file is not None:
                self.file.flush()

        if self.rank == 0:
            for data in self.CALC_DATA:
                if type(data) is str:
                    np.save(self._get_data_path(data), getattr(self, data))

    def save_state(self, i):
        pass

    def load_state(self):
        pass

    def load(self):
        if self.rank == 0:
            for data in self.CALC_DATA:
                if type(data) is str:
                    setattr(self, data, np.load(self._get_data_path(data)))
                else:
                    if self.file is None:
                        self.file = h5py.File("{}.hdf5".format(self.save_path), 'r')

                    data.load(self, self.file)


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

    def write_calc_params(self, params_grp: h5py.Group):
        super().write_calc_params(params_grp)

        params_grp.attrs['dr'] = self.dr
        params_grp.attrs['r_max'] = self.r_max
        params_grp.attrs['Nl'] = self.Nl
        params_grp.attrs['Nr'] = self.sh_grid.Nr

        self.atom_cache.write_params(params_grp)

        if self.uabs_cache is not None:
            self.uabs_cache.write_params(params_grp)
