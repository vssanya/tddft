import tdse
import numpy as np

from .task import CalcData, TaskAtom


class TimeShapeMixin(object):
    def get_shape(self, task):
        return (task.t.size, task.N)

class AzWfArrayData(TimeShapeMixin, CalcData):
    NAME = "az"

    def calc_init(self, task, file):
        super().calc_init(task, file)
        self.az = np.zeros(self.dset.shape[1])

    def calc(self, task, i, t):
        task.calc_field(t)
        tdse.calc_gpu.az(task.wf_array, task.atom_cache, task.E, self.az)

        self.dset[i] = self.az

class WavefuncArrayGPUTask(TaskAtom):
    is_calc_ground_state = True

    N = 10

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

    def save_state(self, i):
        np.save(os.path.join(self.save_path, 'wf_{}.npy'.format(i)), self.wf.asarray())

    def load_state(self, i):
        self.wf.asarray[:] = np.load(os.path.join(self.save_path, 'wf_{}.npy'.format(i)))

    def calc_init(self):
        super().calc_init()

        self.wf_gs = self.calc_ground_state()
        self.wf_array = tdse.wavefunc_gpu.ShWavefuncArrayGPU(self.wf_gs, self.N)

        self.ws = tdse.workspace_gpu.WfArrayGPUWorkspace(self.atom_cache, self.sh_grid, self.uabs_cache, self.N)

        self.E = np.zeros(self.N)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_ground_state(self):
        ws = tdse.workspace.ShWavefuncWS(self.atom_cache, self.sh_grid, self.uabs_cache)
        return tdse.ground_state.wf(self.atom, self.sh_grid, ws, self.dt, 10000)[0]

    def calc_prop(self, i, t):
        self.calc_field(t + self.dt/2)
        self.ws.prop(self.wf_array, self.E, self.dt)

    def calc_field(self, t):
        pass
